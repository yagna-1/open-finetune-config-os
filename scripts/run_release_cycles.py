#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ft_config_engine.db import EngineStore, resolve_database_url
from ft_config_engine.evaluation.harness import run_evaluation
from ft_config_engine.governance import DEFAULT_MODEL_NAME, GovernanceError, ModelGovernanceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated release cycles (evaluate -> staging -> canary)")
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--golden", default="evaluation/golden_dataset.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--prefix", default="cycle")
    parser.add_argument("--promote-production", action="store_true")
    parser.add_argument("--out", default="artifacts/release_cycles/summary.json")
    return parser.parse_args()


def _make_version(prefix: str, index: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{prefix}_{stamp}_{index:02d}"


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    store = EngineStore(resolve_database_url())
    store.create_tables()
    governance = ModelGovernanceService(store)

    cycles: list[dict] = []
    for idx in range(1, max(1, args.cycles) + 1):
        version = _make_version(args.prefix, idx)
        report = run_evaluation(
            candidate_version=version,
            dataset_path=args.dataset,
            golden_path=args.golden,
        )
        report_path = Path("artifacts/eval_reports") / f"{version}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
        gate_payload = [gate.to_dict() for gate in report.gate_results]

        cycle = {
            "version": version,
            "all_gates_passed": bool(report.all_gates_passed),
            "overall_accuracy": report.overall_accuracy,
            "oom_violation_rate": report.oom_violation_rate,
            "ece": report.ece,
            "report_path": str(report_path),
            "state_transitions": [],
        }

        try:
            governance.promote(
                model_name=args.model_name,
                version=version,
                to_state="CANDIDATE",
                actor="release_cycle_runner",
                reason="cycle_evaluated",
                automated=True,
                gate_results={"report_path": str(report_path), "gates": gate_payload},
            )
            cycle["state_transitions"].append("CANDIDATE")
            if report.all_gates_passed:
                governance.promote(
                    model_name=args.model_name,
                    version=version,
                    to_state="STAGING",
                    actor="release_cycle_runner",
                    reason="gates_passed",
                    automated=True,
                    gate_results={"report_path": str(report_path), "gates": gate_payload},
                )
                cycle["state_transitions"].append("STAGING")
                governance.promote(
                    model_name=args.model_name,
                    version=version,
                    to_state="CANARY",
                    actor="release_cycle_runner",
                    reason="staged_for_canary",
                    automated=True,
                    gate_results={"report_path": str(report_path), "gates": gate_payload},
                )
                cycle["state_transitions"].append("CANARY")
                if args.promote_production:
                    governance.promote(
                        model_name=args.model_name,
                        version=version,
                        to_state="PRODUCTION",
                        actor="release_cycle_runner",
                        reason="canary_auto_promote",
                        automated=True,
                        gate_results={"report_path": str(report_path), "gates": gate_payload},
                    )
                    cycle["state_transitions"].append("PRODUCTION")
            else:
                governance.promote(
                    model_name=args.model_name,
                    version=version,
                    to_state="REJECTED",
                    actor="release_cycle_runner",
                    reason="gates_failed",
                    automated=True,
                    gate_results={"report_path": str(report_path), "gates": gate_payload},
                )
                cycle["state_transitions"].append("REJECTED")
        except GovernanceError as exc:
            cycle["governance_error"] = str(exc)

        cycles.append(cycle)

    model_state = governance.list_versions(args.model_name)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "golden": args.golden,
        "model_name": args.model_name,
        "cycles_requested": int(args.cycles),
        "cycles": cycles,
        "final_model_state": model_state,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"release_cycles={len(cycles)}")
    print(f"summary={out_path}")
    print(f"production={model_state.get('production')}")
    print(f"canary={model_state.get('canary')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

