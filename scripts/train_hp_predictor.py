#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ft_config_engine.hp_predictor import (
    save_hyperparameter_predictor,
    train_hyperparameter_predictor,
)
from ft_config_engine.recommender import build_engine_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct hyperparameter predictor")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--out-model", default="artifacts/hp_predictor.joblib")
    parser.add_argument("--out-metrics", default="artifacts/hp_predictor_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_engine_from_dataset(args.dataset, ml_reranker_path=None, hp_predictor_path=None)
    predictor, metadata = train_hyperparameter_predictor(engine.configs, seed=args.seed)
    model_path = save_hyperparameter_predictor(predictor, args.out_model)

    payload = {
        "model_path": str(model_path),
        "dataset_path": args.dataset,
        "normalization_report": engine.normalization_report.to_dict(),
        "metadata": metadata.to_dict(),
    }
    metrics_path = Path(args.out_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"trained_configs={len(engine.configs)}")
    print(f"model_path={model_path}")
    print(f"metrics_path={metrics_path}")
    print(f"trained_targets={','.join(metadata.trained_targets)}")


if __name__ == "__main__":
    main()
