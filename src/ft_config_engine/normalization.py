from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .constants import KNOWN_DATASET_SIZE_BUCKETS, MODEL_SIZE_THRESHOLDS, TASK_ALIASES
from .models import NormalizationReport, NormalizedConfig

_PARAM_COUNT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([BM])\s*$", flags=re.IGNORECASE)


_OPTIMIZER_MAP = {
    "adamw": "adamw_torch",
    "adamw_torch": "adamw_torch",
    "adamw_hf": "adamw_torch",
    "adamw_torch_fused": "adamw_torch",
    "paged_adamw_32bit": "paged_adamw_32bit",
    "paged_adamw_8bit": "paged_adamw_8bit",
    "adafactor": "adafactor",
}

_SCHEDULER_MAP = {
    "cosine": "cosine",
    "linear": "linear",
    "constant": "constant",
    "polynomial": "polynomial",
}

_PRECISION_MAP = {
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp32": "fp32",
    "float32": "fp32",
}


def parse_parameter_count(raw_value: str | int | float) -> int:
    if isinstance(raw_value, (int, float)):
        cast = int(raw_value)
        if cast <= 0:
            raise ValueError(f"unsupported parameter_count format: {raw_value}")
        return cast

    text = str(raw_value).strip()
    if text.isdigit():
        cast = int(text)
        if cast <= 0:
            raise ValueError(f"unsupported parameter_count format: {raw_value}")
        return cast

    match = _PARAM_COUNT_RE.match(text)
    if not match:
        raise ValueError(f"unsupported parameter_count format: {raw_value}")

    value = float(match.group(1))
    scale = match.group(2).upper()
    multiplier = 1_000_000_000 if scale == "B" else 1_000_000
    return int(value * multiplier)


def canonicalize_task_type(raw_task_type: str) -> str:
    cleaned = raw_task_type.strip().lower()
    return TASK_ALIASES.get(cleaned, cleaned)


def bucket_model_size(parameter_count_num: int) -> str:
    if parameter_count_num < MODEL_SIZE_THRESHOLDS["small_max"]:
        return "small"
    if parameter_count_num <= MODEL_SIZE_THRESHOLDS["medium_max"]:
        return "medium"
    return "large"


def bucket_dataset_size(dataset_name: str, dataset_size: int) -> str:
    key = dataset_name.strip().lower()
    if key in KNOWN_DATASET_SIZE_BUCKETS:
        return KNOWN_DATASET_SIZE_BUCKETS[key]

    if dataset_size < 20_000:
        return "small"
    if dataset_size <= 100_000:
        return "medium"
    return "large"


def normalize_optimizer(raw_value: str | None) -> str:
    if not raw_value:
        return "adamw_torch"
    key = str(raw_value).strip().lower()
    return _OPTIMIZER_MAP.get(key, key)


def normalize_scheduler(raw_value: str | None) -> str:
    if not raw_value:
        return "linear"
    key = str(raw_value).strip().lower()
    return _SCHEDULER_MAP.get(key, key)


def normalize_precision(raw_value: str | None) -> str:
    if not raw_value:
        return "fp16"
    key = str(raw_value).strip().lower()
    return _PRECISION_MAP.get(key, "fp16")


def normalize_adapter_type(method: str | None, quantization: str | None) -> str:
    method_key = (method or "none").strip().lower()
    quant_key = (quantization or "").strip().lower()
    if method_key == "none":
        return "none"
    if method_key == "qlora" or quant_key in {"nf4", "fp4", "4bit", "4-bit"}:
        return "qlora"
    return "lora"


def _require_string(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"missing_{field_name}")
    cast = str(value).strip()
    if not cast:
        raise ValueError(f"missing_{field_name}")
    return cast


def _require_positive_int(value: object, field_name: str) -> int:
    if value is None:
        raise ValueError(f"missing_{field_name}")
    cast = int(value)
    if cast <= 0:
        raise ValueError(f"invalid_{field_name}")
    return cast


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    cast = int(value)
    return cast


def _as_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def _first_non_none(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _first_non_empty_string(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        cast = str(value).strip()
        if cast:
            return cast
    return None


def _default_max_seq_length(task_type: str, model_architecture: str, model_name: str) -> int:
    task_key = task_type.strip().lower()
    if task_key in {"instruction_following", "chat", "code_generation", "causal_lm"}:
        return 1024
    if task_key in {"summarization", "translation"}:
        return 1024
    if task_key in {
        "classification",
        "text_classification",
        "sentiment_analysis",
        "question_answering",
        "ner",
        "named_entity_recognition",
        "paraphrase_detection",
    }:
        return 512

    model_key = f"{model_architecture} {model_name}".lower()
    if any(token in model_key for token in {"llama", "mistral", "gemma", "gpt", "qwen", "falcon"}):
        return 1024
    return 512


def normalize_record(raw: dict, line_number: int) -> NormalizedConfig:
    source_raw = raw.get("source")
    source = _as_dict(source_raw)
    task = _as_dict(raw.get("task"))
    dataset = _as_dict(raw.get("dataset"))
    model = _as_dict(raw.get("model"))
    training = _as_dict(raw.get("training_config"))
    adapter = _as_dict(raw.get("adapter_config"))
    lora_config = _as_dict(raw.get("lora_config"))
    hardware = _as_dict(raw.get("hardware"))
    hardware_config = _as_dict(raw.get("hardware_config"))
    performance = _as_dict(raw.get("performance"))

    raw_task_type = _first_non_empty_string(task.get("task_type"), raw.get("task_type"))
    task_type = canonicalize_task_type(_require_string(raw_task_type, "task_type"))
    dataset_name = _require_string(dataset.get("name"), "dataset_name")
    model_name = _require_string(model.get("name"), "model_name")
    parameter_count_raw = _require_string(
        _first_non_empty_string(
            model.get("parameter_count"),
            model.get("size_params"),
            model.get("size"),
        ),
        "model_parameter_count",
    )
    parameter_count_num = parse_parameter_count(parameter_count_raw)

    learning_rate_raw = _first_non_none(training.get("learning_rate"), training.get("lr"))
    if learning_rate_raw is None:
        raise ValueError("missing_learning_rate")
    learning_rate = float(learning_rate_raw)
    if learning_rate <= 0:
        raise ValueError("invalid_learning_rate")

    batch_size_per_device = _require_positive_int(
        _first_non_none(training.get("batch_size_per_device"), training.get("batch_size")),
        "batch_size_per_device",
    )
    gradient_accumulation_steps = _require_positive_int(
        _first_non_none(
            training.get("gradient_accumulation_steps"),
            training.get("grad_accum"),
            1,
        ),
        "gradient_accumulation_steps",
    )

    max_seq_length_value = _first_non_none(training.get("max_seq_length"), raw.get("max_seq_length"))
    if max_seq_length_value is None:
        max_seq_length = _default_max_seq_length(
            task_type=task_type,
            model_architecture=str(model.get("architecture") or "unknown"),
            model_name=model_name,
        )
    else:
        max_seq_length = _require_positive_int(max_seq_length_value, "max_seq_length")

    num_gpus = max(
        1,
        int(
            _first_non_none(
                hardware.get("num_gpus"),
                hardware_config.get("num_gpus"),
                training.get("num_gpus"),
                1,
            )
            or 1
        ),
    )

    effective_batch_size = batch_size_per_device * gradient_accumulation_steps * num_gpus

    dataset_size = _require_positive_int(dataset.get("size"), "dataset_size")
    dataset_size_bucket = bucket_dataset_size(dataset_name=dataset_name, dataset_size=dataset_size)
    model_size_bucket = bucket_model_size(parameter_count_num=parameter_count_num)

    quantization = _first_non_empty_string(
        adapter.get("quantization"),
        hardware.get("quantization"),
        hardware_config.get("quantization"),
    )
    adapter_type = normalize_adapter_type(
        method=_first_non_empty_string(adapter.get("method"), raw.get("adapter")),
        quantization=quantization,
    )
    lora_rank = _optional_int(adapter.get("r")) if adapter_type in {"lora", "qlora"} else None
    if lora_rank is None and adapter_type in {"lora", "qlora"}:
        lora_rank = _optional_int(lora_config.get("r"))

    source_platform = (
        _first_non_empty_string(source.get("platform"), source_raw if isinstance(source_raw, str) else None)
        or "unknown"
    )
    source_url = _first_non_empty_string(source.get("url")) or ""
    lora_alpha = _optional_int(_first_non_none(adapter.get("alpha"), lora_config.get("lora_alpha")))
    lora_dropout_value = _first_non_none(adapter.get("dropout"), lora_config.get("lora_dropout"))
    gpu_type = (
        _first_non_empty_string(hardware.get("gpu_type"), hardware_config.get("target_gpu"))
        or "unknown"
    )
    gpu_memory_gb = float(_first_non_none(hardware.get("gpu_memory_gb"), hardware_config.get("gpu_memory_gb"), 0.0))
    epochs = float(_first_non_none(training.get("epochs"), training.get("num_epochs"), 3.0))

    return NormalizedConfig(
        source_platform=source_platform,
        source_url=source_url,
        record_id=str(raw.get("id") or f"line_{line_number}"),
        task_type=task_type,
        dataset_name=dataset_name,
        dataset_size=dataset_size,
        dataset_size_bucket=dataset_size_bucket,
        model_name=model_name,
        model_architecture=str(model.get("architecture") or "unknown"),
        model_parameter_count=parameter_count_raw,
        model_parameter_count_num=parameter_count_num,
        model_size_bucket=model_size_bucket,
        adapter_type=adapter_type,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size_per_device=batch_size_per_device,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
        optimizer=normalize_optimizer(training.get("optimizer")),
        scheduler=normalize_scheduler(training.get("scheduler")),
        warmup_ratio=float(training.get("warmup_ratio") or 0.0),
        weight_decay=float(training.get("weight_decay") or 0.0),
        max_seq_length=max_seq_length,
        precision=normalize_precision(training.get("precision")),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=float(lora_dropout_value or 0.0)
        if lora_dropout_value is not None
        else None,
        quantization=str(quantization or ""),
        gpu_type=gpu_type,
        gpu_memory_gb=gpu_memory_gb,
        num_gpus=num_gpus,
        performance_metric_name=str(performance.get("metric_name"))
        if performance.get("metric_name") is not None
        else None,
        performance_metric_value=float(performance.get("metric_value"))
        if performance.get("metric_value") is not None
        else None,
        validation_loss=float(performance.get("validation_loss"))
        if performance.get("validation_loss") is not None
        else None,
    )


def deduplicate_configs(configs: list[NormalizedConfig]) -> tuple[list[NormalizedConfig], int]:
    seen: set[tuple] = set()
    deduped: list[NormalizedConfig] = []

    for cfg in configs:
        key = (
            cfg.model_name,
            cfg.dataset_name,
            cfg.task_type,
            cfg.adapter_type,
            cfg.learning_rate,
            cfg.effective_batch_size,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cfg)

    removed = len(configs) - len(deduped)
    return deduped, removed


def load_and_prepare_dataset(path: str | Path) -> tuple[list[NormalizedConfig], NormalizationReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    if path.stat().st_size <= 0:
        return [], NormalizationReport(
            total_rows=0,
            accepted_rows=0,
            rejected_rows=0,
            deduplicated_rows=0,
            rejection_reasons={"dataset_file_empty": 1},
        )

    if _file_is_all_null_bytes(path):
        return [], NormalizationReport(
            total_rows=0,
            accepted_rows=0,
            rejected_rows=0,
            deduplicated_rows=0,
            rejection_reasons={f"dataset_file_null_bytes:{path.name}": 1},
        )

    normalized: list[NormalizedConfig] = []
    rejection_reasons: Counter[str] = Counter()
    total_rows = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                raw = json.loads(line)
            except Exception:  # noqa: BLE001
                rejection_reasons["invalid_json"] += 1
                continue
            try:
                cfg = normalize_record(raw=raw, line_number=line_number)
            except Exception as exc:  # noqa: BLE001
                reason = str(exc).strip() or "invalid_record"
                rejection_reasons[reason] += 1
                continue
            normalized.append(cfg)

    deduped, removed = deduplicate_configs(normalized)

    report = NormalizationReport(
        total_rows=total_rows,
        accepted_rows=len(normalized),
        rejected_rows=total_rows - len(normalized),
        deduplicated_rows=removed,
        rejection_reasons=dict(rejection_reasons),
    )
    return deduped, report


def load_and_prepare_datasets(paths: Iterable[str | Path]) -> tuple[list[NormalizedConfig], NormalizationReport]:
    all_configs: list[NormalizedConfig] = []
    aggregated_reasons: Counter[str] = Counter()
    total_rows = 0
    accepted_rows = 0
    rejected_rows = 0
    deduplicated_rows = 0

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            aggregated_reasons[f"dataset_file_missing:{path.name}"] += 1
            continue

        file_configs, report = load_and_prepare_dataset(path)
        all_configs.extend(file_configs)
        total_rows += report.total_rows
        accepted_rows += report.accepted_rows
        rejected_rows += report.rejected_rows
        deduplicated_rows += report.deduplicated_rows
        aggregated_reasons.update(report.rejection_reasons)

    deduped, cross_dataset_dedup_removed = deduplicate_configs(all_configs)
    deduplicated_rows += cross_dataset_dedup_removed

    merged_report = NormalizationReport(
        total_rows=total_rows,
        accepted_rows=accepted_rows,
        rejected_rows=rejected_rows,
        deduplicated_rows=deduplicated_rows,
        rejection_reasons=dict(aggregated_reasons),
    )
    return deduped, merged_report


def _file_is_all_null_bytes(path: Path) -> bool:
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            if any(byte != 0 for byte in chunk):
                return False
    return True


def save_normalized_jsonl(configs: list[NormalizedConfig], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for cfg in configs:
            handle.write(json.dumps(cfg.to_dict(), ensure_ascii=True) + "\n")
