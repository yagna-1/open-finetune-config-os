from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


AdapterType = str
ModelSizeBucket = str


@dataclass(slots=True)
class NormalizedConfig:
    source_platform: str
    source_url: str
    record_id: str
    task_type: str
    dataset_name: str
    dataset_size: int
    dataset_size_bucket: str
    model_name: str
    model_architecture: str
    model_parameter_count: str
    model_parameter_count_num: int
    model_size_bucket: ModelSizeBucket
    adapter_type: AdapterType
    learning_rate: float
    epochs: float
    batch_size_per_device: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    optimizer: str
    scheduler: str
    warmup_ratio: float
    weight_decay: float
    max_seq_length: int
    precision: str
    lora_rank: int | None
    lora_alpha: int | None
    lora_dropout: float | None
    quantization: str | None
    gpu_type: str
    gpu_memory_gb: float
    num_gpus: int
    performance_metric_name: str | None
    performance_metric_value: float | None
    validation_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StatisticalProfile:
    task_type: str
    model_size_bucket: ModelSizeBucket
    adapter_type: AdapterType
    sample_size: int
    median_learning_rate: float
    learning_rate_q1: float
    learning_rate_q3: float
    learning_rate_iqr: float
    median_effective_batch_size: int
    median_lora_rank: int | None
    typical_optimizer: str
    typical_precision: str
    median_seq_length: int

    def key(self) -> tuple[str, str, str]:
        return (self.task_type, self.model_size_bucket, self.adapter_type)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RecommendationRequest:
    platform: str
    plan: str
    task_type: str
    adapter_type: str
    model_size_bucket: str | None = None
    model_name: str | None = None
    model_parameter_count: str | None = None
    dataset_name: str | None = None
    dataset_size: int | None = None
    sequence_length: int | None = None
    num_gpus: int | None = None
    gpu_override: str | None = None
    epochs: float | None = None
    push_to_hub: bool = False
    huggingface_repo_id: str | None = None
    strategy: str = "auto"
    rerank_top_k: int = 5


@dataclass(slots=True)
class RecommendationResult:
    platform_key: str
    selected_gpu: str
    selected_gpu_memory_gb: float
    safe_hyperparameters: dict[str, Any]
    dependency_stack: list[str]
    estimated_vram_gb_per_gpu: float
    estimated_training_time_hours: float
    recommendation_basis: dict[str, Any]
    notebook_template: str
    notebook_json: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NormalizationReport:
    total_rows: int
    accepted_rows: int
    rejected_rows: int
    deduplicated_rows: int
    rejection_reasons: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
