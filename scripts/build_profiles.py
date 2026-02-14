#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ft_config_engine.normalization import load_and_prepare_datasets, save_normalized_jsonl
from ft_config_engine.statistics import build_statistical_profiles, save_profiles_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized config corpus and statistical profiles")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--out-normalized", default="artifacts/normalized_configs.jsonl")
    parser.add_argument("--out-profiles", default="artifacts/statistical_profiles.json")
    parser.add_argument("--out-report", default="artifacts/normalization_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_paths = [part.strip() for part in str(args.dataset).split(",") if part.strip()]
    configs, report = load_and_prepare_datasets(dataset_paths)
    profiles = build_statistical_profiles(configs)

    save_normalized_jsonl(configs, args.out_normalized)
    save_profiles_json(profiles, args.out_profiles)

    report_path = Path(args.out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"dataset={args.dataset}")
    print(f"normalized_configs={len(configs)}")
    print(f"profiles={len(profiles)}")
    print(f"rejected_rows={report.rejected_rows}")
    print(f"deduplicated_rows={report.deduplicated_rows}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
