#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ft_config_engine.constants import PLATFORM_GPU_MATRIX
from ft_config_engine.models import NormalizedConfig, RecommendationRequest
from ft_config_engine.normalization import load_and_prepare_datasets
from ft_config_engine.recommender import build_engine_from_dataset


TARGET_MIX = {
    "causal_lm": 20,
    "classification": 20,
    "qa": 16,
    "summarization": 12,
    "translation": 10,
    "code": 14,
    "ner": 5,
    "ood_synthetic": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic golden dataset for release-gate evaluation")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--out", default="evaluation/golden_dataset.jsonl")
    return parser.parse_args()


def _category(task_type: str) -> str | None:
    task = task_type.strip().lower()
    if task in {"instruction_following", "chat", "dialogue", "story_generation", "medical_text_generation", "math_reasoning"}:
        return "causal_lm"
    if task in {
        "text_classification",
        "sentiment_analysis",
        "acceptability_classification",
        "question_classification",
        "paraphrase_detection",
        "legal_classification",
        "legal_prediction",
        "paraphrase",
    }:
        return "classification"
    if task in {"question_answering", "medical_qa", "financial_qa", "reading_comprehension"}:
        return "qa"
    if task == "summarization":
        return "summarization"
    if task == "translation":
        return "translation"
    if task in {"code_generation", "code_search", "text2sql"}:
        return "code"
    if task in {"named_entity_recognition", "medical_ner", "relation_extraction"}:
        return "ner"
    return None


def _environment(cfg: NormalizedConfig) -> tuple[str, str, str | None]:
    gpu_type = cfg.gpu_type.strip().upper()
    gpu_mem = float(cfg.gpu_memory_gb)
    if gpu_type in {"T4", "P100"}:
        platform, plan, override = "colab", "free", f"{gpu_type}_16GB"
    elif gpu_type == "V100":
        platform, plan, override = "colab", "pro", "V100_16GB"
    elif gpu_type == "A10G":
        platform, plan, override = "lightning", "pro", "A10G_24GB"
    elif gpu_type == "A100":
        platform, plan, override = "colab", "pro", "A100_40GB"
    elif gpu_mem >= 40:
        platform, plan, override = "colab", "pro", "A100_40GB"
    else:
        platform, plan, override = "colab", "free", "T4_16GB"

    key = f"{platform}_{plan}"
    valid = PLATFORM_GPU_MATRIX.get(key, [])
    if override not in valid:
        override = valid[0] if valid else None
    return platform, plan, override


def _difficulty(index: int, total: int) -> str:
    if index < max(2, total // 3):
        return "easy"
    if index < max(4, (2 * total) // 3):
        return "medium"
    return "hard"


def main() -> int:
    args = parse_args()
    engine = build_engine_from_dataset(Path(args.dataset), ml_reranker_path=None, hp_predictor_path=None)
    configs, _ = load_and_prepare_datasets([Path(args.dataset)])
    pools: dict[str, list[NormalizedConfig]] = defaultdict(list)
    for cfg in configs:
        category = _category(cfg.task_type)
        if category is not None:
            pools[category].append(cfg)

    for category in pools:
        pools[category] = sorted(pools[category], key=lambda item: item.record_id)

    records: list[dict] = []
    record_id = 1
    for category, count in TARGET_MIX.items():
        if category == "ood_synthetic":
            continue
        pool = pools.get(category, [])
        if not pool:
            continue
        i = 0
        attempts = 0
        max_attempts = max(1, len(pool) * 5)
        while i < count and attempts < max_attempts:
            cfg = pool[(i + attempts) % len(pool)]
            attempts += 1
            platform, plan, gpu_override = _environment(cfg)
            lr = float(cfg.learning_rate)
            bs = int(cfg.batch_size_per_device)
            rank = int(cfg.lora_rank or 0)
            request_payload = {
                "task_type": cfg.task_type,
                "model_id": cfg.model_name,
                "model_params": int(cfg.model_parameter_count_num),
                "adapter_type": cfg.adapter_type,
                "dataset_name": cfg.dataset_name,
                "dataset_size": int(cfg.dataset_size),
                "gpu_type": cfg.gpu_type,
                "gpu_vram_gb": float(cfg.gpu_memory_gb),
                "platform": platform,
                "plan": plan,
                "sequence_length": int(cfg.max_seq_length),
                "num_gpus": 1,
                "gpu_override": gpu_override,
                "strategy": "deterministic",
            }
            try:
                engine.recommend(
                    RecommendationRequest(
                        platform=platform,
                        plan=plan,
                        task_type=cfg.task_type,
                        adapter_type=cfg.adapter_type,
                        model_size_bucket=cfg.model_size_bucket,
                        model_name=cfg.model_name,
                        model_parameter_count=str(cfg.model_parameter_count_num),
                        dataset_name=cfg.dataset_name,
                        dataset_size=int(cfg.dataset_size),
                        sequence_length=int(cfg.max_seq_length),
                        num_gpus=1,
                        gpu_override=gpu_override,
                        strategy="deterministic",
                    ),
                    render_notebook=False,
                )
            except Exception:
                continue
            records.append(
                {
                    "id": f"golden_{record_id:03d}",
                    "category": category,
                    "difficulty": _difficulty(i, count),
                    "request": request_payload,
                    "ground_truth": {
                        "learning_rate": lr,
                        "per_device_train_batch_size": bs,
                        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                        "lora_r": int(cfg.lora_rank) if cfg.lora_rank is not None else None,
                        "known_safe_on_gpu": True,
                    },
                    "acceptable_ranges": {
                        "learning_rate": [max(1e-6, lr * 0.1), min(1e-2, lr * 10.0)],
                        "batch_size": [1, max(16, bs * 16)],
                        "lora_r": None if rank == 0 else [4, 128],
                    },
                }
            )
            record_id += 1
            i += 1

    ood_rows = [
        {
            "task_type": "totally_unknown_task_family",
            "model_id": "meta-llama/Llama-2-7b-hf",
            "model_params": 7000000000,
            "adapter_type": "qlora",
            "dataset_name": "unknown_dataset",
            "dataset_size": 10000,
            "gpu_type": "T4",
            "gpu_vram_gb": 16.0,
            "platform": "colab",
            "plan": "free",
            "sequence_length": 1024,
            "num_gpus": 1,
            "gpu_override": "T4_16GB",
            "strategy": "deterministic",
        },
        {
            "task_type": "x_bio_ner_future",
            "model_id": "google/gemma-7b",
            "model_params": 7000000000,
            "adapter_type": "qlora",
            "dataset_name": "synthetic_bio",
            "dataset_size": 5000,
            "gpu_type": "T4",
            "gpu_vram_gb": 16.0,
            "platform": "colab",
            "plan": "free",
            "sequence_length": 1024,
            "num_gpus": 1,
            "gpu_override": "T4_16GB",
            "strategy": "deterministic",
        },
        {
            "task_type": "robotics_dialog_policy",
            "model_id": "mistralai/Mistral-7B-v0.1",
            "model_params": 7000000000,
            "adapter_type": "qlora",
            "dataset_name": "robotics_policy",
            "dataset_size": 8000,
            "gpu_type": "T4",
            "gpu_vram_gb": 16.0,
            "platform": "colab",
            "plan": "free",
            "sequence_length": 1024,
            "num_gpus": 1,
            "gpu_override": "T4_16GB",
            "strategy": "deterministic",
        },
    ]
    for row in ood_rows:
        records.append(
            {
                "id": f"golden_{record_id:03d}",
                "category": "ood_synthetic",
                "difficulty": "hard",
                "request": row,
                "ground_truth": {
                    "learning_rate": 2e-4,
                    "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 4,
                    "lora_r": 16,
                    "known_safe_on_gpu": True,
                },
                "acceptable_ranges": {
                    "learning_rate": [1e-5, 5e-4],
                    "batch_size": [1, 8],
                    "lora_r": [4, 64],
                },
            }
        )
        record_id += 1

    if len(records) > 100:
        records = records[:100]
    elif len(records) < 100 and records:
        cursor = 0
        while len(records) < 100:
            base = dict(records[cursor % len(records)])
            base["id"] = f"golden_{record_id:03d}"
            base["difficulty"] = "hard"
            records.append(base)
            record_id += 1
            cursor += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"golden_rows={len(records)}")
    print(f"output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
