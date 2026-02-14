#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from ft_config_engine.db import EngineStore, resolve_database_url
from ft_config_engine.governance import DEFAULT_MODEL_NAME, ModelGovernanceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a model version across governance states")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--to-state",
        required=True,
        choices=["CANDIDATE", "STAGING", "CANARY", "PRODUCTION", "DEPRECATED", "REJECTED"],
    )
    parser.add_argument("--actor", default="manual_operator")
    parser.add_argument("--reason", default="")
    parser.add_argument("--automated", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = EngineStore(resolve_database_url())
    store.create_tables()
    governance = ModelGovernanceService(store)
    result = governance.promote(
        model_name=args.model,
        version=args.version,
        to_state=args.to_state,
        actor=args.actor,
        reason=args.reason,
        automated=args.automated,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

