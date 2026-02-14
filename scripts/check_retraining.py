#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ft_config_engine.db import EngineStore, resolve_database_url
from ft_config_engine.governance import ModelGovernanceService
from ft_config_engine.retraining import RetrainingScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and optionally execute retraining trigger")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--run", action="store_true", help="Execute training/evaluation pipeline if eligible")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = EngineStore(resolve_database_url())
    store.create_tables()
    governance = ModelGovernanceService(store)
    scheduler = RetrainingScheduler(store=store, governance=governance)
    result = scheduler.trigger_if_needed(dataset=args.dataset, run_training=args.run)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

