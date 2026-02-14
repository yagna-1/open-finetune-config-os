from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .models import NormalizedConfig, StatisticalProfile


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median received empty values")
    midpoint = n // 2
    if n % 2 == 1:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile received empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    return float(
        ordered[low] + (ordered[high] - ordered[low]) * (index - low),
    )


def _mode(values: list[str], default: str) -> str:
    if not values:
        return default
    counts = Counter(values)
    max_count = max(counts.values())
    winners = sorted([key for key, count in counts.items() if count == max_count])
    return winners[0]


def build_statistical_profiles(
    configs: list[NormalizedConfig],
) -> dict[tuple[str, str, str], StatisticalProfile]:
    grouped: dict[tuple[str, str, str], list[NormalizedConfig]] = defaultdict(list)
    for cfg in configs:
        grouped[(cfg.task_type, cfg.model_size_bucket, cfg.adapter_type)].append(cfg)

    profiles: dict[tuple[str, str, str], StatisticalProfile] = {}
    for key, rows in grouped.items():
        learning_rates = [row.learning_rate for row in rows]
        effective_batches = [float(row.effective_batch_size) for row in rows]
        seq_lengths = [float(row.max_seq_length) for row in rows]
        lora_ranks = [float(row.lora_rank) for row in rows if row.lora_rank is not None]
        optimizers = [row.optimizer for row in rows]
        precisions = [row.precision for row in rows]

        q1 = _percentile(learning_rates, 0.25)
        q3 = _percentile(learning_rates, 0.75)
        median_lora_rank: int | None = None
        if lora_ranks:
            median_lora_rank = int(round(_median(lora_ranks)))

        profile = StatisticalProfile(
            task_type=key[0],
            model_size_bucket=key[1],
            adapter_type=key[2],
            sample_size=len(rows),
            median_learning_rate=_median(learning_rates),
            learning_rate_q1=q1,
            learning_rate_q3=q3,
            learning_rate_iqr=q3 - q1,
            median_effective_batch_size=int(round(_median(effective_batches))),
            median_lora_rank=median_lora_rank,
            typical_optimizer=_mode(optimizers, default="adamw_torch"),
            typical_precision=_mode(precisions, default="fp16"),
            median_seq_length=int(round(_median(seq_lengths))),
        )
        profiles[key] = profile

    return profiles


def save_profiles_json(
    profiles: dict[tuple[str, str, str], StatisticalProfile],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "|".join(key): profile.to_dict()
        for key, profile in sorted(profiles.items(), key=lambda item: item[0])
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
