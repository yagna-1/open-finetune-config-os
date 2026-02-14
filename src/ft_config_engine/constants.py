from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_gb: float
    speed_factor: float
    tokens_per_second: float
    supports_bf16: bool


PLATFORM_GPU_MATRIX: dict[str, list[str]] = {
    "colab_free": ["T4_16GB", "P100_16GB"],
    "colab_pro": ["T4_16GB", "V100_16GB", "A100_40GB"],
    "kaggle_free": ["T4_16GB", "P100_16GB"],
    "lightning_free": ["CPU"],
    "lightning_pro": ["A10G_24GB", "A100_40GB"],
}


GPU_SPECS: dict[str, GPUSpec] = {
    "CPU": GPUSpec(
        name="CPU",
        memory_gb=0.0,
        speed_factor=0.03,
        tokens_per_second=120.0,
        supports_bf16=False,
    ),
    "T4_16GB": GPUSpec(
        name="NVIDIA T4 16GB",
        memory_gb=16.0,
        speed_factor=1.0,
        tokens_per_second=1500.0,
        supports_bf16=False,
    ),
    "P100_16GB": GPUSpec(
        name="NVIDIA P100 16GB",
        memory_gb=16.0,
        speed_factor=1.15,
        tokens_per_second=1800.0,
        supports_bf16=False,
    ),
    "V100_16GB": GPUSpec(
        name="NVIDIA V100 16GB",
        memory_gb=16.0,
        speed_factor=1.9,
        tokens_per_second=2900.0,
        supports_bf16=False,
    ),
    "V100_32GB": GPUSpec(
        name="NVIDIA V100 32GB",
        memory_gb=32.0,
        speed_factor=2.1,
        tokens_per_second=3400.0,
        supports_bf16=False,
    ),
    "A10G_24GB": GPUSpec(
        name="NVIDIA A10G 24GB",
        memory_gb=24.0,
        speed_factor=2.4,
        tokens_per_second=3800.0,
        supports_bf16=True,
    ),
    "A100_40GB": GPUSpec(
        name="NVIDIA A100 40GB",
        memory_gb=40.0,
        speed_factor=4.0,
        tokens_per_second=6500.0,
        supports_bf16=True,
    ),
    "A100_80GB": GPUSpec(
        name="NVIDIA A100 80GB",
        memory_gb=80.0,
        speed_factor=5.0,
        tokens_per_second=7600.0,
        supports_bf16=True,
    ),
    "RTX_3090_24GB": GPUSpec(
        name="NVIDIA RTX 3090 24GB",
        memory_gb=24.0,
        speed_factor=2.0,
        tokens_per_second=3200.0,
        supports_bf16=False,
    ),
    "RTX_4090_24GB": GPUSpec(
        name="NVIDIA RTX 4090 24GB",
        memory_gb=24.0,
        speed_factor=2.8,
        tokens_per_second=4400.0,
        supports_bf16=True,
    ),
}


KNOWN_DATASET_SIZE_BUCKETS: dict[str, str] = {
    "openassistant": "small",
    "alpaca": "medium",
    "dolly": "small",
    "sharegpt": "large",
    "squad": "small",
    "cnn_dailymail": "large",
    "xsum": "medium",
    "wikitext": "medium",
    "ag_news": "medium",
    "imdb": "medium",
    "yelp_polarity": "large",
    "imagenet_1k": "large",
    "code_alpaca": "small",
}


DEPENDENCY_MATRIX: dict[str, list[str]] = {
    "colab": [
        "torch==2.1.2",
        "transformers==4.38.2",
        "peft==0.10.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.28.0",
        "datasets==2.18.0",
    ],
    "kaggle": [
        "torch==2.1.2",
        "transformers==4.38.2",
        "peft==0.10.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.28.0",
        "datasets==2.18.0",
    ],
    "lightning": [
        "torch==2.1.2",
        "transformers==4.38.2",
        "peft==0.10.0",
        "bitsandbytes==0.43.1",
        "accelerate==0.28.0",
        "datasets==2.18.0",
    ],
}


NOTEBOOK_DATASET_ALIASES: dict[str, str] = {
    "alpaca": "yahma/alpaca-cleaned",
    "alpaca-cleaned": "yahma/alpaca-cleaned",
    "dolly": "databricks/databricks-dolly-15k",
    "dolly-15k": "databricks/databricks-dolly-15k",
    "guanaco-llama2-1k": "timdettmers/openassistant-guanaco",
    "openassistant-guanaco": "timdettmers/openassistant-guanaco",
}


NOTEBOOK_MODEL_ALIASES: dict[str, str] = {
    "meta-llama/llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "meta-llama/llama-2-13b-hf": "meta-llama/Llama-2-13b-hf",
    "mistralai/mistral-7b-v0.1": "mistralai/Mistral-7B-v0.1",
}


NOTEBOOK_GATED_MODEL_PREFIXES: tuple[str, ...] = (
    "meta-llama/",
    "google/gemma",
    "mistralai/",
)


TASK_ALIASES: dict[str, str] = {
    "language finetuning": "instruction_following",
    "language_finetuning": "instruction_following",
    "finetuning": "instruction_following",
    "fine_tuning": "instruction_following",
    "instruction_tuning": "instruction_following",
    "sft": "instruction_following",
    "supervised_finetuning": "instruction_following",
    "causal_lm": "instruction_following",
    "chatbot": "chat",
    "conversation": "chat",
    "conversational": "chat",
    "qa": "question_answering",
    "ocr": "optical_character_recognition",
    "ocr_finetuning": "optical_character_recognition",
    "ocr_fine_tuning": "optical_character_recognition",
    "asr_finetuning": "speech_recognition",
    "speech_finetuning": "speech_recognition",
}


MODEL_SIZE_THRESHOLDS = {
    "small_max": 1_000_000_000,  # <1B
    "medium_max": 7_000_000_000,  # 1B-7B
}


RECOMMENDATION_MIN_SAMPLE = 3
