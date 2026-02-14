from __future__ import annotations

from .constants import GPU_SPECS


def _precision_bytes(precision: str) -> float:
    key = precision.lower().strip()
    if key == "fp32":
        return 4.0
    return 2.0


def max_safe_lora_rank(gpu_memory_gb: float) -> int:
    if gpu_memory_gb >= 80:
        return 128
    if gpu_memory_gb >= 40:
        return 64
    if gpu_memory_gb >= 24:
        return 32
    if gpu_memory_gb >= 16:
        return 16
    return 8


def max_safe_seq_length(gpu_memory_gb: float) -> int:
    if gpu_memory_gb >= 80:
        return 4096
    if gpu_memory_gb >= 40:
        return 2048
    if gpu_memory_gb >= 24:
        return 1536
    if gpu_memory_gb >= 16:
        return 1024
    return 512


def estimate_training_vram_gb_per_gpu(
    parameter_count: int,
    adapter_type: str,
    precision: str,
    batch_size_per_device: int,
    sequence_length: int,
    num_gpus: int,
    lora_rank: int | None = None,
) -> float:
    if num_gpus <= 0:
        num_gpus = 1

    adapter_key = adapter_type.lower().strip()
    if adapter_key == "qlora":
        bytes_per_param = 0.5
    else:
        bytes_per_param = _precision_bytes(precision)

    model_memory_bytes = parameter_count * bytes_per_param

    if adapter_key == "none":
        optimizer_multiplier = 2.0
    elif adapter_key == "lora":
        optimizer_multiplier = 0.25
    else:
        optimizer_multiplier = 0.10

    if adapter_key in {"lora", "qlora"} and lora_rank:
        rank_scale = max(0.5, min(2.0, lora_rank / 16.0))
        optimizer_multiplier *= rank_scale

    optimizer_bytes = model_memory_bytes * optimizer_multiplier

    activation_scale = max(0.5, (batch_size_per_device * sequence_length) / 1024.0)
    activation_bytes = model_memory_bytes * 0.30 * activation_scale

    total_bytes = (model_memory_bytes + optimizer_bytes + activation_bytes) * 1.20
    per_gpu_bytes = total_bytes / num_gpus
    gb = per_gpu_bytes / (1024**3)
    return float(round(gb, 4))


def estimate_training_time_hours(
    dataset_rows: int,
    sequence_length: int,
    epochs: float,
    effective_batch_size: int,
    selected_gpu: str,
    num_gpus: int,
    adapter_type: str,
) -> float:
    rows = max(1, dataset_rows)
    eff_batch = max(1, effective_batch_size)
    seq_len = max(64, sequence_length)
    gpu_count = max(1, num_gpus)

    gpu = GPU_SPECS[selected_gpu]

    adapter_multiplier = {
        "none": 0.85,
        "lora": 1.00,
        "qlora": 0.92,
    }.get(adapter_type, 1.0)

    steps = (rows * max(epochs, 0.1)) / eff_batch
    tokens_per_step = seq_len * eff_batch
    total_tokens = steps * tokens_per_step

    throughput = gpu.tokens_per_second * gpu_count * adapter_multiplier
    if throughput <= 0:
        return 0.0

    hours = total_tokens / throughput / 3600.0
    return float(round(hours, 3))
