from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from .memory import estimate_training_vram_gb_per_gpu
from .models import NormalizedConfig, RecommendationRequest

try:  # pragma: no cover - exercised via runtime import checks
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:  # pragma: no cover
    joblib = None
    pd = None
    ColumnTransformer = None
    Ridge = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


MODEL_VERSION = "1.0.0"
MAX_QUERIES_PER_GROUP = 128
MAX_CANDIDATES_PER_QUERY = 64

_LOWER_IS_BETTER_METRICS = {
    "perplexity",
    "validation_loss",
    "loss",
    "wer",
}

NUMERIC_FEATURES = [
    "query_sequence_length",
    "query_num_gpus",
    "query_gpu_memory_gb",
    "candidate_log10_params",
    "candidate_log10_dataset_size",
    "candidate_effective_batch_size",
    "candidate_max_seq_length",
    "candidate_log10_learning_rate",
    "candidate_lora_rank",
    "candidate_gpu_memory_gb",
    "candidate_validation_loss",
    "candidate_metric_value",
    "same_model",
    "same_dataset",
    "seq_relative_gap",
    "lr_relative_gap",
    "estimated_vram_gb_per_gpu",
    "vram_utilization_ratio",
    "candidate_quality_prior",
    "candidate_has_validation_loss",
    "candidate_has_metric",
]

CATEGORICAL_FEATURES = [
    "query_task_type",
    "query_model_size_bucket",
    "query_adapter_type",
    "query_model_name",
    "query_dataset_name",
    "candidate_model_name",
    "candidate_dataset_name",
    "candidate_optimizer",
    "candidate_scheduler",
    "candidate_precision",
    "candidate_adapter_type",
    "candidate_model_size_bucket",
    "candidate_metric_name",
]


@dataclass(slots=True)
class MLRerankerMetadata:
    model_version: str
    trained_at_utc: str
    seed: int
    train_rows: int
    validation_rows: int
    train_queries: int
    validation_queries: int
    dataset_configs: int
    metrics: dict[str, float]
    numeric_features: list[str]
    categorical_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLReranker:
    def __init__(self, pipeline: Any, metadata: MLRerankerMetadata) -> None:
        self.pipeline = pipeline
        self.metadata = metadata

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    def predict_scores(
        self,
        request: RecommendationRequest,
        task_type: str,
        model_size_bucket: str,
        adapter_type: str,
        candidates: list[NormalizedConfig],
        gpu_memory_gb: float,
    ) -> dict[str, float]:
        _require_runtime_dependencies()
        if not candidates:
            return {}

        rows = [
            _build_pair_features_for_request(
                request=request,
                task_type=task_type,
                model_size_bucket=model_size_bucket,
                adapter_type=adapter_type,
                candidate=candidate,
                gpu_memory_gb=gpu_memory_gb,
            )
            for candidate in candidates
        ]
        frame = pd.DataFrame(rows)
        scores = self.pipeline.predict(frame)
        return {
            candidate.record_id: float(score)
            for candidate, score in zip(candidates, scores, strict=True)
        }


def train_ml_reranker(
    configs: list[NormalizedConfig],
    *,
    seed: int = 42,
) -> tuple[MLReranker, MLRerankerMetadata]:
    _require_runtime_dependencies()
    if len(configs) < 32:
        raise ValueError("insufficient configs to train ml reranker")

    rows, labels, query_ids = _build_training_dataset(configs, seed=seed)
    if len(rows) < 64:
        raise ValueError("insufficient training rows to train ml reranker")

    unique_queries = sorted(set(query_ids))
    rng = Random(seed)
    rng.shuffle(unique_queries)

    split_index = max(1, int(round(len(unique_queries) * 0.8)))
    if split_index >= len(unique_queries):
        split_index = len(unique_queries) - 1

    train_query_ids = set(unique_queries[:split_index])
    valid_query_ids = set(unique_queries[split_index:])
    if not valid_query_ids:
        valid_query_ids = set(unique_queries[-1:])
        train_query_ids = set(unique_queries[:-1])

    train_indices = [idx for idx, query_id in enumerate(query_ids) if query_id in train_query_ids]
    valid_indices = [idx for idx, query_id in enumerate(query_ids) if query_id in valid_query_ids]
    if not train_indices or not valid_indices:
        raise ValueError("unable to create train/validation split for ml reranker")

    frame = pd.DataFrame(rows)
    y = labels

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    model = Ridge(alpha=1.0)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    train_frame = frame.iloc[train_indices]
    valid_frame = frame.iloc[valid_indices]
    y_train = [y[idx] for idx in train_indices]
    y_valid = [y[idx] for idx in valid_indices]

    pipeline.fit(train_frame, y_train)
    valid_pred = pipeline.predict(valid_frame)

    rmse = math.sqrt(
        sum((pred - target) ** 2 for pred, target in zip(valid_pred, y_valid, strict=True)) / len(y_valid)
    )
    mae = sum(abs(pred - target) for pred, target in zip(valid_pred, y_valid, strict=True)) / len(y_valid)

    ndcg_at_5, top1_regret, oom_violation_rate = _ranking_metrics_by_query(
        query_ids=[query_ids[idx] for idx in valid_indices],
        predictions=list(valid_pred),
        labels=y_valid,
        vram_utilization=[float(valid_frame.iloc[idx]["vram_utilization_ratio"]) for idx in range(len(valid_frame))],
    )

    metadata = MLRerankerMetadata(
        model_version=MODEL_VERSION,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        train_rows=len(train_indices),
        validation_rows=len(valid_indices),
        train_queries=len(train_query_ids),
        validation_queries=len(valid_query_ids),
        dataset_configs=len(configs),
        metrics={
            "rmse": round(float(rmse), 6),
            "mae": round(float(mae), 6),
            "ndcg_at_5": round(float(ndcg_at_5), 6),
            "top1_regret": round(float(top1_regret), 6),
            "oom_violation_rate": round(float(oom_violation_rate), 6),
        },
        numeric_features=NUMERIC_FEATURES[:],
        categorical_features=CATEGORICAL_FEATURES[:],
    )
    return MLReranker(pipeline=pipeline, metadata=metadata), metadata


def save_ml_reranker(ml_reranker: MLReranker, output_path: str | Path) -> Path:
    _require_runtime_dependencies()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": ml_reranker.metadata.to_dict(),
        "pipeline": ml_reranker.pipeline,
    }
    joblib.dump(payload, path)
    return path


def load_ml_reranker(path: str | Path) -> tuple[MLReranker | None, str]:
    if _IMPORT_ERROR is not None:
        return None, f"ml-reranker-unavailable:{type(_IMPORT_ERROR).__name__}"

    resolved = Path(path).expanduser()
    if not resolved.exists():
        return None, f"ml-reranker-missing:{resolved}"

    try:
        payload = joblib.load(resolved)
    except Exception as exc:  # noqa: BLE001
        return None, f"ml-reranker-load-failed:{type(exc).__name__}"

    metadata_raw = payload.get("metadata") if isinstance(payload, dict) else None
    pipeline = payload.get("pipeline") if isinstance(payload, dict) else None
    if not isinstance(metadata_raw, dict) or pipeline is None:
        return None, "ml-reranker-invalid-payload"

    try:
        metadata = MLRerankerMetadata(
            model_version=str(metadata_raw.get("model_version") or MODEL_VERSION),
            trained_at_utc=str(metadata_raw.get("trained_at_utc") or ""),
            seed=int(metadata_raw.get("seed") or 0),
            train_rows=int(metadata_raw.get("train_rows") or 0),
            validation_rows=int(metadata_raw.get("validation_rows") or 0),
            train_queries=int(metadata_raw.get("train_queries") or 0),
            validation_queries=int(metadata_raw.get("validation_queries") or 0),
            dataset_configs=int(metadata_raw.get("dataset_configs") or 0),
            metrics={str(k): float(v) for k, v in dict(metadata_raw.get("metrics") or {}).items()},
            numeric_features=[str(item) for item in list(metadata_raw.get("numeric_features") or [])],
            categorical_features=[str(item) for item in list(metadata_raw.get("categorical_features") or [])],
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"ml-reranker-invalid-metadata:{type(exc).__name__}"

    return MLReranker(pipeline=pipeline, metadata=metadata), "ml-reranker-loaded"


def _build_training_dataset(
    configs: list[NormalizedConfig],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[float], list[str]]:
    grouped: dict[tuple[str, str, str], list[NormalizedConfig]] = {}
    for cfg in configs:
        key = (cfg.task_type, cfg.model_size_bucket, cfg.adapter_type)
        grouped.setdefault(key, []).append(cfg)

    rng = Random(seed)
    rows: list[dict[str, Any]] = []
    labels: list[float] = []
    query_ids: list[str] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        queries = group[:]
        rng.shuffle(queries)
        queries = queries[:MAX_QUERIES_PER_GROUP]
        for query in queries:
            candidates = group[:]
            rng.shuffle(candidates)
            candidates = candidates[:MAX_CANDIDATES_PER_QUERY]
            if query not in candidates:
                candidates = candidates[:-1] + [query] if candidates else [query]
            for candidate in candidates:
                row = _build_pair_features_for_query(query=query, candidate=candidate)
                label = _pair_relevance_score(query=query, candidate=candidate)
                rows.append(row)
                labels.append(label)
                query_ids.append(query.record_id)
    return rows, labels, query_ids


def _build_pair_features_for_query(
    *,
    query: NormalizedConfig,
    candidate: NormalizedConfig,
) -> dict[str, Any]:
    query_seq = max(128, query.max_seq_length)
    query_num_gpus = max(1, query.num_gpus)
    query_gpu_memory = max(0.0, query.gpu_memory_gb)

    est_vram = estimate_training_vram_gb_per_gpu(
        parameter_count=candidate.model_parameter_count_num,
        adapter_type=candidate.adapter_type,
        precision=candidate.precision,
        batch_size_per_device=candidate.batch_size_per_device,
        sequence_length=min(query_seq, candidate.max_seq_length),
        num_gpus=query_num_gpus,
        lora_rank=candidate.lora_rank,
    )
    return _pair_feature_row(
        query_task_type=query.task_type,
        query_model_size_bucket=query.model_size_bucket,
        query_adapter_type=query.adapter_type,
        query_model_name=query.model_name,
        query_dataset_name=query.dataset_name,
        query_sequence_length=query_seq,
        query_num_gpus=query_num_gpus,
        query_gpu_memory_gb=query_gpu_memory,
        query_learning_rate=query.learning_rate,
        candidate=candidate,
        est_vram=est_vram,
    )


def _build_pair_features_for_request(
    *,
    request: RecommendationRequest,
    task_type: str,
    model_size_bucket: str,
    adapter_type: str,
    candidate: NormalizedConfig,
    gpu_memory_gb: float,
) -> dict[str, Any]:
    query_seq = max(128, int(request.sequence_length or 1024))
    query_num_gpus = max(1, int(request.num_gpus or 1))
    query_gpu_memory = max(0.0, float(gpu_memory_gb))

    est_vram = estimate_training_vram_gb_per_gpu(
        parameter_count=candidate.model_parameter_count_num,
        adapter_type=candidate.adapter_type,
        precision=candidate.precision,
        batch_size_per_device=candidate.batch_size_per_device,
        sequence_length=min(query_seq, candidate.max_seq_length),
        num_gpus=query_num_gpus,
        lora_rank=candidate.lora_rank,
    )
    return _pair_feature_row(
        query_task_type=task_type,
        query_model_size_bucket=model_size_bucket,
        query_adapter_type=adapter_type,
        query_model_name=request.model_name or "",
        query_dataset_name=request.dataset_name or "",
        query_sequence_length=query_seq,
        query_num_gpus=query_num_gpus,
        query_gpu_memory_gb=query_gpu_memory,
        query_learning_rate=None,
        candidate=candidate,
        est_vram=est_vram,
    )


def _pair_feature_row(
    *,
    query_task_type: str,
    query_model_size_bucket: str,
    query_adapter_type: str,
    query_model_name: str,
    query_dataset_name: str,
    query_sequence_length: int,
    query_num_gpus: int,
    query_gpu_memory_gb: float,
    query_learning_rate: float | None,
    candidate: NormalizedConfig,
    est_vram: float,
) -> dict[str, Any]:
    seq_gap = abs(float(candidate.max_seq_length) - float(query_sequence_length)) / max(
        float(candidate.max_seq_length),
        float(query_sequence_length),
        1.0,
    )
    lr_gap = 0.0
    if query_learning_rate and query_learning_rate > 0 and candidate.learning_rate > 0:
        lr_gap = min(
            1.0,
            abs(math.log10(candidate.learning_rate) - math.log10(query_learning_rate)) / 3.0,
        )

    query_model_norm = (query_model_name or "").strip().lower()
    query_dataset_norm = (query_dataset_name or "").strip().lower()
    candidate_model_norm = candidate.model_name.strip().lower()
    candidate_dataset_norm = candidate.dataset_name.strip().lower()

    quality_prior = _outcome_quality(candidate)
    metric_value = (
        float(candidate.performance_metric_value)
        if candidate.performance_metric_value is not None
        else 0.0
    )
    validation_loss = (
        float(candidate.validation_loss)
        if candidate.validation_loss is not None
        else 0.0
    )
    vram_utilization = est_vram / query_gpu_memory_gb if query_gpu_memory_gb > 0 else 0.0

    return {
        "query_task_type": query_task_type,
        "query_model_size_bucket": query_model_size_bucket,
        "query_adapter_type": query_adapter_type,
        "query_model_name": query_model_name,
        "query_dataset_name": query_dataset_name,
        "query_sequence_length": float(query_sequence_length),
        "query_num_gpus": float(query_num_gpus),
        "query_gpu_memory_gb": float(query_gpu_memory_gb),
        "candidate_model_name": candidate.model_name,
        "candidate_dataset_name": candidate.dataset_name,
        "candidate_optimizer": candidate.optimizer,
        "candidate_scheduler": candidate.scheduler,
        "candidate_precision": candidate.precision,
        "candidate_adapter_type": candidate.adapter_type,
        "candidate_model_size_bucket": candidate.model_size_bucket,
        "candidate_metric_name": (candidate.performance_metric_name or "").lower(),
        "candidate_log10_params": math.log10(max(1.0, float(candidate.model_parameter_count_num))),
        "candidate_log10_dataset_size": math.log10(max(1.0, float(candidate.dataset_size))),
        "candidate_effective_batch_size": float(candidate.effective_batch_size),
        "candidate_max_seq_length": float(candidate.max_seq_length),
        "candidate_log10_learning_rate": math.log10(max(candidate.learning_rate, 1e-10)),
        "candidate_lora_rank": float(candidate.lora_rank or 0),
        "candidate_gpu_memory_gb": float(candidate.gpu_memory_gb),
        "candidate_validation_loss": validation_loss,
        "candidate_metric_value": metric_value,
        "same_model": 1.0 if query_model_norm and query_model_norm == candidate_model_norm else 0.0,
        "same_dataset": 1.0 if query_dataset_norm and query_dataset_norm == candidate_dataset_norm else 0.0,
        "seq_relative_gap": float(max(0.0, min(1.0, seq_gap))),
        "lr_relative_gap": float(max(0.0, min(1.0, lr_gap))),
        "estimated_vram_gb_per_gpu": float(est_vram),
        "vram_utilization_ratio": float(max(0.0, vram_utilization)),
        "candidate_quality_prior": float(quality_prior),
        "candidate_has_validation_loss": 1.0 if candidate.validation_loss is not None else 0.0,
        "candidate_has_metric": 1.0 if candidate.performance_metric_value is not None else 0.0,
    }


def _pair_relevance_score(query: NormalizedConfig, candidate: NormalizedConfig) -> float:
    model_match = 1.0 if query.model_name == candidate.model_name else 0.0
    dataset_match = 1.0 if query.dataset_name == candidate.dataset_name else 0.0

    seq_gap = abs(float(candidate.max_seq_length) - float(query.max_seq_length)) / max(
        float(candidate.max_seq_length),
        float(query.max_seq_length),
        1.0,
    )
    seq_compat = 1.0 - max(0.0, min(1.0, seq_gap))

    lr_gap = abs(math.log10(candidate.learning_rate) - math.log10(query.learning_rate)) / 3.0
    lr_compat = 1.0 - max(0.0, min(1.0, lr_gap))

    compatibility = (0.45 * model_match) + (0.20 * dataset_match) + (0.20 * seq_compat) + (0.15 * lr_compat)
    quality = _outcome_quality(candidate)
    base = (0.55 * compatibility) + (0.45 * quality)
    if query.record_id == candidate.record_id:
        base = min(1.0, base + 0.10)
    return float(max(0.0, min(1.0, base)))


def _outcome_quality(cfg: NormalizedConfig) -> float:
    values: list[float] = []
    if cfg.validation_loss is not None and cfg.validation_loss > 0:
        values.append(max(0.0, min(1.0, 1.0 / (1.0 + cfg.validation_loss))))

    if cfg.performance_metric_name and cfg.performance_metric_value is not None:
        metric_name = cfg.performance_metric_name.strip().lower()
        metric_value = float(cfg.performance_metric_value)
        if metric_name in _LOWER_IS_BETTER_METRICS:
            values.append(max(0.0, min(1.0, 1.0 / (1.0 + max(metric_value, 0.0)))))
        else:
            if metric_value > 1.0:
                metric_value /= 100.0
            values.append(max(0.0, min(1.0, metric_value)))

    if not values:
        return 0.5
    return float(sum(values) / len(values))


def _ranking_metrics_by_query(
    *,
    query_ids: list[str],
    predictions: list[float],
    labels: list[float],
    vram_utilization: list[float],
) -> tuple[float, float, float]:
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for query_id, pred, label, vram in zip(
        query_ids,
        predictions,
        labels,
        vram_utilization,
        strict=True,
    ):
        grouped.setdefault(query_id, []).append((float(pred), float(label), float(vram)))

    ndcgs: list[float] = []
    regrets: list[float] = []
    oom_flags: list[float] = []
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda item: item[0], reverse=True)
        labels_ranked = [item[1] for item in ranked]
        ideal = sorted(labels_ranked, reverse=True)
        ndcgs.append(_ndcg_at_k(labels_ranked, ideal, k=5))

        best_label = max(labels_ranked) if labels_ranked else 0.0
        top_label = labels_ranked[0] if labels_ranked else 0.0
        regrets.append(max(0.0, best_label - top_label))

        top_vram = ranked[0][2] if ranked else 0.0
        oom_flags.append(1.0 if top_vram > 1.0 else 0.0)

    if not ndcgs:
        return 0.0, 0.0, 0.0

    return (
        float(sum(ndcgs) / len(ndcgs)),
        float(sum(regrets) / len(regrets)),
        float(sum(oom_flags) / len(oom_flags)),
    )


def _ndcg_at_k(labels_ranked: list[float], labels_ideal: list[float], k: int) -> float:
    def dcg(values: list[float]) -> float:
        score = 0.0
        for idx, value in enumerate(values[:k], start=1):
            score += (2.0**value - 1.0) / math.log2(idx + 1.0)
        return score

    ranked_dcg = dcg(labels_ranked)
    ideal_dcg = dcg(labels_ideal)
    if ideal_dcg <= 0:
        return 0.0
    return float(ranked_dcg / ideal_dcg)


def _require_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"ml reranker dependencies missing: {_IMPORT_ERROR}") from _IMPORT_ERROR
