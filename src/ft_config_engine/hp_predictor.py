from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any

from .models import NormalizedConfig

try:  # pragma: no cover - exercised via runtime import checks
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception as exc:  # pragma: no cover
    joblib = None
    pd = None
    ColumnTransformer = None
    RandomForestRegressor = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None
    _IMPORT_ERROR = exc
else:  # pragma: no cover
    _IMPORT_ERROR = None


MODEL_VERSION = "1.0.0"

NUMERIC_FEATURES = [
    "log10_model_params",
    "gpu_memory_gb",
    "num_gpus",
    "log10_dataset_size",
    "requested_sequence_length",
]

CATEGORICAL_FEATURES = [
    "task_type",
    "model_size_bucket",
    "adapter_type",
    "dataset_size_bucket",
    "model_architecture",
]

TARGET_KEYS = [
    "log10_learning_rate",
    "log2_effective_batch_size",
    "max_seq_length",
    "lora_rank",
]

TARGET_SCALES = {
    "log10_learning_rate": 2.0,
    "log2_effective_batch_size": 5.0,
    "max_seq_length": 768.0,
    "lora_rank": 48.0,
}


@dataclass(slots=True)
class HyperparameterPredictorMetadata:
    model_version: str
    trained_at_utc: str
    seed: int
    train_rows: int
    validation_rows: int
    dataset_configs: int
    trained_targets: list[str]
    target_metrics: dict[str, dict[str, float]]
    numeric_features: list[str]
    categorical_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HyperparameterPredictor:
    def __init__(
        self,
        models: dict[str, Any],
        target_uncertainty_p90: dict[str, float],
        metadata: HyperparameterPredictorMetadata,
    ) -> None:
        self.models = models
        self.target_uncertainty_p90 = target_uncertainty_p90
        self.metadata = metadata

    @property
    def model_version(self) -> str:
        return self.metadata.model_version

    def predict(
        self,
        *,
        task_type: str,
        model_size_bucket: str,
        adapter_type: str,
        model_architecture: str,
        model_parameter_count: int,
        dataset_size: int,
        dataset_size_bucket: str,
        gpu_memory_gb: float,
        num_gpus: int,
        requested_sequence_length: int,
    ) -> dict[str, dict[str, float]]:
        _require_runtime_dependencies()

        row = _feature_row(
            task_type=task_type,
            model_size_bucket=model_size_bucket,
            adapter_type=adapter_type,
            model_architecture=model_architecture,
            model_parameter_count=model_parameter_count,
            dataset_size=dataset_size,
            dataset_size_bucket=dataset_size_bucket,
            gpu_memory_gb=gpu_memory_gb,
            num_gpus=num_gpus,
            requested_sequence_length=requested_sequence_length,
        )
        frame = pd.DataFrame([row])

        predictions: dict[str, dict[str, float]] = {}
        for target_key, model in self.models.items():
            raw = float(model.predict(frame)[0])
            uncertainty = float(self.target_uncertainty_p90.get(target_key, 0.0))
            confidence = _target_confidence(target_key=target_key, uncertainty=uncertainty)
            predictions[target_key] = _decode_prediction(
                target_key=target_key,
                raw_prediction=raw,
                raw_uncertainty=uncertainty,
                confidence=confidence,
            )
        return predictions


def train_hyperparameter_predictor(
    configs: list[NormalizedConfig],
    *,
    seed: int = 42,
) -> tuple[HyperparameterPredictor, HyperparameterPredictorMetadata]:
    _require_runtime_dependencies()
    if len(configs) < 64:
        raise ValueError("insufficient configs to train hyperparameter predictor")

    rows = [_feature_row_from_config(cfg) for cfg in configs]
    frame = pd.DataFrame(rows)

    indices = list(range(len(frame)))
    rng = Random(seed)
    rng.shuffle(indices)
    split = max(1, int(round(len(indices) * 0.8)))
    if split >= len(indices):
        split = len(indices) - 1
    train_idx = indices[:split]
    valid_idx = indices[split:]
    if not train_idx or not valid_idx:
        raise ValueError("unable to create train/validation split for hyperparameter predictor")

    train_frame = frame.iloc[train_idx]
    valid_frame = frame.iloc[valid_idx]

    models: dict[str, Any] = {}
    uncertainty_p90: dict[str, float] = {}
    target_metrics: dict[str, dict[str, float]] = {}
    trained_targets: list[str] = []

    for target_key in TARGET_KEYS:
        if target_key not in train_frame.columns:
            continue

        target_train = train_frame[target_key]
        target_valid = valid_frame[target_key]

        # Only train LoRA rank on adapter-aware rows with rank labels.
        if target_key == "lora_rank":
            train_mask = train_frame["adapter_type"].isin({"lora", "qlora"}) & train_frame[target_key].notna()
            valid_mask = valid_frame["adapter_type"].isin({"lora", "qlora"}) & valid_frame[target_key].notna()
            if int(train_mask.sum()) < 24 or int(valid_mask.sum()) < 8:
                continue
            model_train = train_frame.loc[train_mask]
            model_valid = valid_frame.loc[valid_mask]
            y_train = target_train.loc[train_mask]
            y_valid = target_valid.loc[valid_mask]
        else:
            if target_train.notna().sum() < 32 or target_valid.notna().sum() < 8:
                continue
            model_train = train_frame[target_train.notna()]
            model_valid = valid_frame[target_valid.notna()]
            y_train = target_train[target_train.notna()]
            y_valid = target_valid[target_valid.notna()]

        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        model = RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=2,
            random_state=seed + len(trained_targets),
            n_jobs=1,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model),
            ]
        )

        pipeline.fit(model_train, y_train)
        pred_valid = pipeline.predict(model_valid)
        errors = [abs(float(pred) - float(actual)) for pred, actual in zip(pred_valid, y_valid, strict=True)]
        mae = sum(errors) / len(errors)
        p90 = _percentile(errors, 0.90)

        models[target_key] = pipeline
        uncertainty_p90[target_key] = float(p90)
        target_metrics[target_key] = {
            "mae": round(float(mae), 6),
            "p90_abs_error": round(float(p90), 6),
            "validation_rows": float(len(errors)),
        }
        trained_targets.append(target_key)

    if not models:
        raise ValueError("no valid targets were trainable for hyperparameter predictor")

    metadata = HyperparameterPredictorMetadata(
        model_version=MODEL_VERSION,
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        train_rows=len(train_idx),
        validation_rows=len(valid_idx),
        dataset_configs=len(configs),
        trained_targets=trained_targets,
        target_metrics=target_metrics,
        numeric_features=NUMERIC_FEATURES[:],
        categorical_features=CATEGORICAL_FEATURES[:],
    )
    predictor = HyperparameterPredictor(
        models=models,
        target_uncertainty_p90=uncertainty_p90,
        metadata=metadata,
    )
    return predictor, metadata


def save_hyperparameter_predictor(
    predictor: HyperparameterPredictor,
    output_path: str | Path,
) -> Path:
    _require_runtime_dependencies()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": predictor.metadata.to_dict(),
        "models": predictor.models,
        "target_uncertainty_p90": predictor.target_uncertainty_p90,
    }
    joblib.dump(payload, path)
    return path


def load_hyperparameter_predictor(path: str | Path) -> tuple[HyperparameterPredictor | None, str]:
    if _IMPORT_ERROR is not None:
        return None, f"hp-predictor-unavailable:{type(_IMPORT_ERROR).__name__}"

    resolved = Path(path).expanduser()
    if not resolved.exists():
        return None, f"hp-predictor-missing:{resolved}"

    try:
        payload = joblib.load(resolved)
    except Exception as exc:  # noqa: BLE001
        return None, f"hp-predictor-load-failed:{type(exc).__name__}"

    if not isinstance(payload, dict):
        return None, "hp-predictor-invalid-payload"
    metadata_raw = payload.get("metadata")
    models = payload.get("models")
    uncertainty = payload.get("target_uncertainty_p90")

    if not isinstance(metadata_raw, dict) or not isinstance(models, dict) or not isinstance(uncertainty, dict):
        return None, "hp-predictor-invalid-payload"
    if not models:
        return None, "hp-predictor-empty-models"

    try:
        metadata = HyperparameterPredictorMetadata(
            model_version=str(metadata_raw.get("model_version") or MODEL_VERSION),
            trained_at_utc=str(metadata_raw.get("trained_at_utc") or ""),
            seed=int(metadata_raw.get("seed") or 0),
            train_rows=int(metadata_raw.get("train_rows") or 0),
            validation_rows=int(metadata_raw.get("validation_rows") or 0),
            dataset_configs=int(metadata_raw.get("dataset_configs") or 0),
            trained_targets=[str(item) for item in list(metadata_raw.get("trained_targets") or [])],
            target_metrics={
                str(key): {str(metric): float(value) for metric, value in dict(metrics).items()}
                for key, metrics in dict(metadata_raw.get("target_metrics") or {}).items()
            },
            numeric_features=[str(item) for item in list(metadata_raw.get("numeric_features") or [])],
            categorical_features=[str(item) for item in list(metadata_raw.get("categorical_features") or [])],
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"hp-predictor-invalid-metadata:{type(exc).__name__}"

    predictor = HyperparameterPredictor(
        models={str(key): model for key, model in models.items()},
        target_uncertainty_p90={str(key): float(value) for key, value in uncertainty.items()},
        metadata=metadata,
    )
    return predictor, "hp-predictor-loaded"


def _feature_row_from_config(cfg: NormalizedConfig) -> dict[str, Any]:
    row = _feature_row(
        task_type=cfg.task_type,
        model_size_bucket=cfg.model_size_bucket,
        adapter_type=cfg.adapter_type,
        model_architecture=cfg.model_architecture,
        model_parameter_count=cfg.model_parameter_count_num,
        dataset_size=cfg.dataset_size,
        dataset_size_bucket=cfg.dataset_size_bucket,
        gpu_memory_gb=cfg.gpu_memory_gb,
        num_gpus=cfg.num_gpus,
        requested_sequence_length=cfg.max_seq_length,
    )
    row["log10_learning_rate"] = math.log10(max(cfg.learning_rate, 1e-10))
    row["log2_effective_batch_size"] = math.log2(max(float(cfg.effective_batch_size), 1.0))
    row["max_seq_length"] = float(cfg.max_seq_length)
    row["lora_rank"] = float(cfg.lora_rank) if cfg.lora_rank is not None else math.nan
    return row


def _feature_row(
    *,
    task_type: str,
    model_size_bucket: str,
    adapter_type: str,
    model_architecture: str,
    model_parameter_count: int,
    dataset_size: int,
    dataset_size_bucket: str,
    gpu_memory_gb: float,
    num_gpus: int,
    requested_sequence_length: int,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "model_size_bucket": model_size_bucket,
        "adapter_type": adapter_type,
        "dataset_size_bucket": dataset_size_bucket,
        "model_architecture": model_architecture,
        "log10_model_params": math.log10(max(float(model_parameter_count), 1.0)),
        "gpu_memory_gb": float(max(0.0, gpu_memory_gb)),
        "num_gpus": float(max(1, num_gpus)),
        "log10_dataset_size": math.log10(max(float(dataset_size), 1.0)),
        "requested_sequence_length": float(max(128, requested_sequence_length)),
    }


def _target_confidence(target_key: str, uncertainty: float) -> float:
    scale = TARGET_SCALES.get(target_key, 1.0)
    if scale <= 0:
        return 0.5
    ratio = max(0.0, uncertainty / scale)
    return max(0.05, min(0.95, 1.0 - ratio))


def _decode_prediction(
    *,
    target_key: str,
    raw_prediction: float,
    raw_uncertainty: float,
    confidence: float,
) -> dict[str, float]:
    low_raw = raw_prediction - raw_uncertainty
    high_raw = raw_prediction + raw_uncertainty

    if target_key == "log10_learning_rate":
        value = 10.0**raw_prediction
        low = 10.0**low_raw
        high = 10.0**high_raw
    elif target_key == "log2_effective_batch_size":
        value = 2.0**raw_prediction
        low = 2.0**low_raw
        high = 2.0**high_raw
    else:
        value = raw_prediction
        low = low_raw
        high = high_raw

    return {
        "value": float(value),
        "low": float(low),
        "high": float(high),
        "confidence": float(confidence),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * max(0.0, min(1.0, quantile))))
    return float(ordered[idx])


def _require_runtime_dependencies() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(f"hyperparameter predictor dependencies missing: {_IMPORT_ERROR}") from _IMPORT_ERROR

