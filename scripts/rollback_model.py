#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from ft_config_engine.db import EngineStore, resolve_database_url
from ft_config_engine.governance import DEFAULT_MODEL_NAME, GovernanceError, ModelGovernanceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergency model rollback")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--to-version", default=None)
    parser.add_argument("--reason", default="emergency_rollback")
    parser.add_argument("--actor", default="on_call_engineer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = EngineStore(resolve_database_url())
    store.create_tables()
    governance = ModelGovernanceService(store)
    try:
        result = governance.rollback(
            model_name=args.model,
            to_version=args.to_version,
            actor=args.actor,
            reason=args.reason,
        )
    except GovernanceError as exc:
        print(f"rollback_failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

