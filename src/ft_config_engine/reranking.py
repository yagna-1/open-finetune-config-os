from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .memory import estimate_training_vram_gb_per_gpu
from .models import NormalizedConfig, RecommendationRequest

_LOWER_IS_BETTER_METRICS = {
    "perplexity",
    "validation_loss",
    "loss",
    "wer",
}


@dataclass(slots=True)
class RankedCandidate:
    config: NormalizedConfig
    total_score: float
    heuristic_total_score: float | None
    ml_score: float | None
    similarity_score: float
    performance_score: float
    efficiency_score: float
    estimated_vram_gb_per_gpu: float
    score_breakdown: dict[str, float]


def rank_candidates(
    candidates: list[NormalizedConfig],
    request: RecommendationRequest,
    gpu_memory_gb: float,
    num_gpus: int,
    sequence_length: int,
) -> list[RankedCandidate]:
    if not candidates:
        return []

    val_loss_values = [row.validation_loss for row in candidates if row.validation_loss and row.validation_loss > 0]
    metric_groups: dict[str, list[float]] = {}
    effective_batches = [float(row.effective_batch_size) for row in candidates]
    min_batch = min(effective_batches)
    max_batch = max(effective_batches)

    for row in candidates:
        if row.performance_metric_name and row.performance_metric_value is not None:
            metric_groups.setdefault(row.performance_metric_name, []).append(row.performance_metric_value)

    ranked: list[RankedCandidate] = []
    for row in candidates:
        seq_target = max(128, sequence_length)
        est_vram = estimate_training_vram_gb_per_gpu(
            parameter_count=row.model_parameter_count_num,
            adapter_type=row.adapter_type,
            precision=row.precision,
            batch_size_per_device=row.batch_size_per_device,
            sequence_length=min(seq_target, row.max_seq_length),
            num_gpus=max(1, num_gpus),
            lora_rank=row.lora_rank,
        )

        model_score = 0.7
        if request.model_name:
            model_score = 1.0 if row.model_name == request.model_name else 0.25

        dataset_score = 0.65
        if request.dataset_name:
            dataset_score = 1.0 if row.dataset_name == request.dataset_name else 0.30

        seq_alignment = 1.0 - (
            abs(float(row.max_seq_length) - float(seq_target))
            / max(float(row.max_seq_length), float(seq_target), 1.0)
        )
        seq_alignment = _clamp(seq_alignment)

        gpu_alignment = 1.0 - (
            abs(float(row.gpu_memory_gb) - float(gpu_memory_gb))
            / max(float(row.gpu_memory_gb), float(gpu_memory_gb), 1.0)
        )
        gpu_alignment = _clamp(gpu_alignment)

        similarity_score = (
            0.35 * model_score
            + 0.25 * dataset_score
            + 0.20 * seq_alignment
            + 0.20 * gpu_alignment
        )

        val_loss_score = 0.5
        if row.validation_loss is not None and row.validation_loss > 0 and len(val_loss_values) > 1:
            val_loss_score = _min_max_score(
                value=row.validation_loss,
                values=val_loss_values,
                lower_is_better=True,
            )

        metric_score = 0.5
        if row.performance_metric_name and row.performance_metric_value is not None:
            metric_values = metric_groups.get(row.performance_metric_name, [])
            if len(metric_values) > 1:
                metric_score = _min_max_score(
                    value=row.performance_metric_value,
                    values=metric_values,
                    lower_is_better=row.performance_metric_name.lower() in _LOWER_IS_BETTER_METRICS,
                )
            else:
                metric_score = _single_metric_score(
                    metric_name=row.performance_metric_name,
                    metric_value=row.performance_metric_value,
                )

        performance_score = 0.60 * val_loss_score + 0.40 * metric_score

        if gpu_memory_gb > 0:
            memory_headroom = _clamp((gpu_memory_gb - est_vram) / gpu_memory_gb)
        else:
            memory_headroom = 0.5

        batch_score = 0.5
        if max_batch > min_batch:
            batch_score = _clamp((row.effective_batch_size - min_batch) / (max_batch - min_batch))

        efficiency_score = 0.70 * memory_headroom + 0.30 * batch_score

        total_score = 0.40 * similarity_score + 0.40 * performance_score + 0.20 * efficiency_score
        if gpu_memory_gb > 0 and est_vram > gpu_memory_gb:
            total_score *= 0.60

        ranked.append(
            RankedCandidate(
                config=row,
                total_score=round(total_score, 6),
                heuristic_total_score=round(total_score, 6),
                ml_score=None,
                similarity_score=round(similarity_score, 6),
                performance_score=round(performance_score, 6),
                efficiency_score=round(efficiency_score, 6),
                estimated_vram_gb_per_gpu=round(est_vram, 4),
                score_breakdown={
                    "model_score": round(model_score, 6),
                    "dataset_score": round(dataset_score, 6),
                    "seq_alignment": round(seq_alignment, 6),
                    "gpu_alignment": round(gpu_alignment, 6),
                    "val_loss_score": round(val_loss_score, 6),
                    "metric_score": round(metric_score, 6),
                    "memory_headroom": round(memory_headroom, 6),
                    "batch_score": round(batch_score, 6),
                },
            )
        )

    ranked.sort(key=lambda item: item.total_score, reverse=True)
    return ranked


def rerank_with_ml_scores(
    ranked: list[RankedCandidate],
    ml_scores: dict[str, float],
    ml_weight: float = 0.65,
) -> list[RankedCandidate]:
    if not ranked:
        return []
    ml_weight = _clamp(ml_weight)
    heuristic_weight = 1.0 - ml_weight

    heuristic_values = [candidate.total_score for candidate in ranked]
    low = min(heuristic_values)
    high = max(heuristic_values)

    reranked: list[RankedCandidate] = []
    for candidate in ranked:
        heuristic_norm = _normalize_min_max(candidate.total_score, low, high)
        ml_raw = float(ml_scores.get(candidate.config.record_id, 0.5))
        ml_clamped = _clamp(ml_raw)
        final_score = (heuristic_weight * heuristic_norm) + (ml_weight * ml_clamped)
        breakdown = dict(candidate.score_breakdown)
        breakdown["heuristic_score"] = round(heuristic_norm, 6)
        breakdown["ml_score"] = round(ml_clamped, 6)
        breakdown["final_blended_score"] = round(final_score, 6)

        reranked.append(
            RankedCandidate(
                config=candidate.config,
                total_score=round(final_score, 6),
                heuristic_total_score=round(candidate.total_score, 6),
                ml_score=round(ml_clamped, 6),
                similarity_score=candidate.similarity_score,
                performance_score=candidate.performance_score,
                efficiency_score=candidate.efficiency_score,
                estimated_vram_gb_per_gpu=candidate.estimated_vram_gb_per_gpu,
                score_breakdown=breakdown,
            )
        )

    reranked.sort(key=lambda item: item.total_score, reverse=True)
    return reranked


def aggregate_from_ranked_candidates(
    ranked: list[RankedCandidate],
    top_k: int,
    fallback_pool: list[NormalizedConfig],
) -> dict[str, float | int | str | None]:
    if not ranked:
        return _aggregate_fallback(fallback_pool)

    top = ranked[: max(1, top_k)]
    weights = _normalized_weights([max(1e-6, candidate.total_score) for candidate in top])

    learning_rate = _weighted_mean([candidate.config.learning_rate for candidate in top], weights)
    effective_batch = int(round(_weighted_mean([candidate.config.effective_batch_size for candidate in top], weights)))
    max_seq_length = int(round(_weighted_mean([candidate.config.max_seq_length for candidate in top], weights)))

    lora_values = [candidate.config.lora_rank for candidate in top if candidate.config.lora_rank is not None]
    lora_rank: int | None = None
    if lora_values:
        lora_rank = int(round(median(lora_values)))

    return {
        "learning_rate": learning_rate,
        "effective_batch_size": max(1, effective_batch),
        "max_seq_length": max(128, max_seq_length),
        "optimizer": _weighted_mode([candidate.config.optimizer for candidate in top], weights, default="adamw_torch"),
        "scheduler": _weighted_mode([candidate.config.scheduler for candidate in top], weights, default="linear"),
        "precision": _weighted_mode([candidate.config.precision for candidate in top], weights, default="fp16"),
        "lora_rank": lora_rank,
        "model_name": _weighted_mode([candidate.config.model_name for candidate in top], weights, default=""),
        "dataset_name": _weighted_mode([candidate.config.dataset_name for candidate in top], weights, default=""),
    }


def ranked_candidates_summary(ranked: list[RankedCandidate], top_n: int = 5) -> list[dict]:
    payload: list[dict] = []
    for candidate in ranked[: max(1, top_n)]:
        payload.append(
            {
                "record_id": candidate.config.record_id,
                "model_name": candidate.config.model_name,
                "dataset_name": candidate.config.dataset_name,
                "learning_rate": candidate.config.learning_rate,
                "effective_batch_size": candidate.config.effective_batch_size,
                "lora_rank": candidate.config.lora_rank,
                "total_score": candidate.total_score,
                "heuristic_total_score": candidate.heuristic_total_score,
                "ml_score": candidate.ml_score,
                "similarity_score": candidate.similarity_score,
                "performance_score": candidate.performance_score,
                "efficiency_score": candidate.efficiency_score,
                "estimated_vram_gb_per_gpu": candidate.estimated_vram_gb_per_gpu,
            }
        )
    return payload


def _aggregate_fallback(pool: list[NormalizedConfig]) -> dict[str, float | int | str | None]:
    if not pool:
        return {
            "learning_rate": 2e-4,
            "effective_batch_size": 16,
            "max_seq_length": 1024,
            "optimizer": "adamw_torch",
            "scheduler": "linear",
            "precision": "fp16",
            "lora_rank": None,
            "model_name": "",
            "dataset_name": "",
        }
    return {
        "learning_rate": float(median([row.learning_rate for row in pool])),
        "effective_batch_size": int(round(median([row.effective_batch_size for row in pool]))),
        "max_seq_length": int(round(median([row.max_seq_length for row in pool]))),
        "optimizer": pool[0].optimizer,
        "scheduler": pool[0].scheduler,
        "precision": pool[0].precision,
        "lora_rank": int(round(median([row.lora_rank for row in pool if row.lora_rank is not None])))
        if any(row.lora_rank is not None for row in pool)
        else None,
        "model_name": pool[0].model_name,
        "dataset_name": pool[0].dataset_name,
    }


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights, strict=True)))


def _weighted_mode(values: list[str], weights: list[float], default: str) -> str:
    if not values:
        return default
    scores: dict[str, float] = {}
    for value, weight in zip(values, weights, strict=True):
        scores[value] = scores.get(value, 0.0) + weight
    best = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return best or default


def _normalized_weights(scores: list[float]) -> list[float]:
    total = sum(scores)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [score / total for score in scores]


def _single_metric_score(metric_name: str, metric_value: float) -> float:
    name = metric_name.lower()
    if name in _LOWER_IS_BETTER_METRICS:
        return _clamp(1.0 / (1.0 + max(metric_value, 0.0)))
    return _clamp(metric_value if metric_value <= 1.0 else metric_value / 100.0)


def _min_max_score(value: float, values: list[float], lower_is_better: bool) -> float:
    low = min(values)
    high = max(values)
    if high == low:
        return 0.5
    if lower_is_better:
        return _clamp((high - value) / (high - low))
    return _clamp((value - low) / (high - low))


def _normalize_min_max(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return _clamp((value - low) / (high - low))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))
