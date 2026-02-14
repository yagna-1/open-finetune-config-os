#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ft_config_engine.db import EngineStore, resolve_database_url
from ft_config_engine.evaluation.harness import run_evaluation
from ft_config_engine.governance import DEFAULT_MODEL_NAME, GovernanceError, ModelGovernanceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a candidate model/recommendation version")
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--dataset", default="finetuning_configs_final.jsonl")
    parser.add_argument("--golden", default="evaluation/golden_dataset.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--auto-promote-to-staging", action="store_true")
    parser.add_argument("--out-dir", default="artifacts/eval_reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Running evaluation for candidate v{args.candidate_version}...")
    report = run_evaluation(
        candidate_version=args.candidate_version,
        dataset_path=args.dataset,
        golden_path=args.golden,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{args.candidate_version}.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")

    print("=" * 60)
    print(f"EVALUATION RESULT: {'PASS' if report.all_gates_passed else 'FAIL'}")
    print("=" * 60)
    for gate in report.gate_results:
        status = "PASS" if gate.passed else "FAIL"
        print(f"[{status}] {gate.gate_name}: {gate.detail}")
    print(f"Report saved to: {report_path}")

    store = EngineStore(resolve_database_url())
    store.create_tables()
    governance = ModelGovernanceService(store)
    gate_payload = [gate.to_dict() for gate in report.gate_results]
    try:
        governance.promote(
            model_name=args.model_name,
            version=args.candidate_version,
            to_state="CANDIDATE",
            actor="automated_pipeline",
            reason="candidate_evaluated",
            automated=True,
            gate_results={"report_path": str(report_path), "gates": gate_payload},
        )
    except GovernanceError:
        # Candidate state may already exist; continue.
        pass

    if report.all_gates_passed and args.auto_promote_to_staging:
        governance.promote_to_staging(
            version=args.candidate_version,
            model_name=args.model_name,
            gate_results={"report_path": str(report_path), "gates": gate_payload},
        )
        print(f"v{args.candidate_version} promoted to STAGING.")
    elif not report.all_gates_passed:
        governance.promote(
            model_name=args.model_name,
            version=args.candidate_version,
            to_state="REJECTED",
            actor="automated_pipeline",
            reason="evaluation_gates_failed",
            automated=True,
            gate_results={"report_path": str(report_path), "gates": gate_payload},
        )
        print("Candidate rejected because one or more release gates failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
