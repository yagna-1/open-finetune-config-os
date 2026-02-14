#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ft_config_engine.ml_ranker import save_ml_reranker, train_ml_reranker
from ft_config_engine.recommender import build_engine_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML reranker for hybrid_ml strategy")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--out-model", default="artifacts/ml_reranker.joblib")
    parser.add_argument("--out-metrics", default="artifacts/ml_reranker_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = build_engine_from_dataset(args.dataset, ml_reranker_path=None)
    reranker, metadata = train_ml_reranker(engine.configs, seed=args.seed)
    model_path = save_ml_reranker(reranker, args.out_model)

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
    print(f"ndcg_at_5={metadata.metrics.get('ndcg_at_5', 0.0)}")
    print(f"top1_regret={metadata.metrics.get('top1_regret', 0.0)}")
    print(f"oom_violation_rate={metadata.metrics.get('oom_violation_rate', 0.0)}")


if __name__ == "__main__":
    main()
