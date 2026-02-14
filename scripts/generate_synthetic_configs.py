#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from ft_config_engine.normalization import load_and_prepare_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic sparse-cell fine-tuning configurations")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--out", default="artifacts/synthetic_configs.jsonl")
    parser.add_argument("--per-cell", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _clamp_lr(value: float) -> float:
    return max(1e-6, min(1e-2, value))


def _nearest_lora_rank(value: int) -> int:
    allowed = [4, 8, 16, 32, 64, 128]
    return min(allowed, key=lambda candidate: abs(candidate - value))


def _nearest_pow2(value: int) -> int:
    value = max(1, min(512, value))
    return max(1, 2 ** int(round(math.log2(value))))


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    configs, _ = load_and_prepare_datasets([Path(args.dataset)])
    by_cell: dict[tuple[str, str, str], list] = {}
    for cfg in configs:
        by_cell.setdefault((cfg.task_type, cfg.model_size_bucket, cfg.adapter_type), []).append(cfg)

    cells = list(by_cell.keys())
    synthetic_rows: list[dict] = []
    for cell in cells:
        pool = by_cell[cell]
        if len(pool) >= args.per_cell:
            continue
        needed = args.per_cell - len(pool)
        for i in range(needed):
            left = random.choice(pool)
            right = random.choice(pool)
            alpha = random.random()
            lr = _clamp_lr(alpha * left.learning_rate + (1.0 - alpha) * right.learning_rate)
            bs = _nearest_pow2(int(round(alpha * left.batch_size_per_device + (1.0 - alpha) * right.batch_size_per_device)))
            grad = max(1, int(round(alpha * left.gradient_accumulation_steps + (1.0 - alpha) * right.gradient_accumulation_steps)))
            eff = bs * grad * max(1, left.num_gpus)
            rank = None
            if left.lora_rank is not None or right.lora_rank is not None:
                raw_rank = int(round(alpha * float(left.lora_rank or 16) + (1.0 - alpha) * float(right.lora_rank or 16)))
                rank = _nearest_lora_rank(raw_rank)

            payload = left.to_dict()
            payload.update(
                {
                    "record_id": f"synthetic::{cell[0]}::{cell[1]}::{cell[2]}::{i:04d}",
                    "learning_rate": lr,
                    "batch_size_per_device": bs,
                    "gradient_accumulation_steps": grad,
                    "effective_batch_size": eff,
                    "lora_rank": rank,
                    "source_platform": "synthetic",
                    "source_url": "synthetic://interpolated",
                }
            )
            synthetic_rows.append(payload)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in synthetic_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"synthetic_rows={len(synthetic_rows)}")
    print(f"output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

