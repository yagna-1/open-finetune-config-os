#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ft_config_engine.models import RecommendationRequest
from ft_config_engine.recommender import build_engine_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic fine-tuning recommendation")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--ml-reranker", dest="ml_reranker_path")
    parser.add_argument("--hp-predictor", dest="hp_predictor_path")
    parser.add_argument("--platform", required=True, choices=["colab", "kaggle", "lightning"])
    parser.add_argument("--plan", required=True, choices=["free", "pro"])
    parser.add_argument("--task", required=True, dest="task_type")
    parser.add_argument("--adapter", required=True, choices=["none", "lora", "qlora"], dest="adapter_type")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], dest="model_size_bucket")
    parser.add_argument("--model-name")
    parser.add_argument("--model-params", dest="model_parameter_count")
    parser.add_argument("--dataset-name")
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--gpu-override")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", dest="huggingface_repo_id")
    parser.add_argument("--strategy", choices=["auto", "hybrid", "deterministic", "hybrid_ml"], default="auto")
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--out-json", default="artifacts/recommendation.json")
    parser.add_argument("--out-notebook", default="artifacts/recommended_notebook.ipynb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_engine_from_dataset(
        args.dataset,
        ml_reranker_path=args.ml_reranker_path,
        hp_predictor_path=args.hp_predictor_path,
    )
    req = RecommendationRequest(
        platform=args.platform,
        plan=args.plan,
        task_type=args.task_type,
        adapter_type=args.adapter_type,
        model_size_bucket=args.model_size_bucket,
        model_name=args.model_name,
        model_parameter_count=args.model_parameter_count,
        dataset_name=args.dataset_name,
        dataset_size=args.dataset_size,
        sequence_length=args.sequence_length,
        num_gpus=args.num_gpus,
        gpu_override=args.gpu_override,
        epochs=args.epochs,
        push_to_hub=args.push_to_hub,
        huggingface_repo_id=args.huggingface_repo_id,
        strategy=args.strategy,
        rerank_top_k=args.rerank_top_k,
    )
    result = engine.recommend(req, render_notebook=True)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    if result.notebook_json is not None:
        out_notebook = Path(args.out_notebook)
        out_notebook.parent.mkdir(parents=True, exist_ok=True)
        out_notebook.write_text(
            json.dumps(result.notebook_json, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    print(f"saved recommendation: {out_json}")
    print(f"saved notebook: {args.out_notebook}")
    print(f"selected_gpu: {result.selected_gpu}")
    print(f"estimated_vram_gb_per_gpu: {result.estimated_vram_gb_per_gpu}")


if __name__ == "__main__":
    main()
