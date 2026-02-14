from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .db import EngineStore
from .governance import DEFAULT_MODEL_NAME, ModelGovernanceService


@dataclass(slots=True)
class RetrainingPolicy:
    cooldown_hours: int = 72
    volume_threshold: int = 50
    oom_window_days: int = 7
    min_events_for_oom_rate: int = 20
    oom_rate_threshold: float = 0.03


@dataclass(slots=True)
class RetrainingDecision:
    should_retrain: bool
    reason: str
    success_count_since: int
    window_total_events: int
    window_oom_events: int
    window_oom_rate: float
    unprocessed_events: int
    last_trained_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrainingScheduler:
    def __init__(
        self,
        store: EngineStore,
        governance: ModelGovernanceService,
        *,
        state_path: str | Path = "artifacts/retraining_state.json",
        policy: RetrainingPolicy | None = None,
    ) -> None:
        self.store = store
        self.governance = governance
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or RetrainingPolicy()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, payload: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def evaluate(self) -> RetrainingDecision:
        now = datetime.now(timezone.utc)
        state = self._load_state()
        last_trained_raw = state.get("last_trained_at")
        last_trained_at = None
        if isinstance(last_trained_raw, str):
            try:
                last_trained_at = datetime.fromisoformat(last_trained_raw)
            except Exception:
                last_trained_at = None

        if last_trained_at is not None:
            elapsed = now - last_trained_at
            if elapsed < timedelta(hours=self.policy.cooldown_hours):
                snapshot = self.store.telemetry_trigger_snapshot(
                    success_since=last_trained_at,
                    window_start=now - timedelta(days=self.policy.oom_window_days),
                )
                return RetrainingDecision(
                    should_retrain=False,
                    reason=f"cooldown_active:{elapsed.total_seconds() / 3600:.2f}h",
                    success_count_since=int(snapshot["success_count_since"]),
                    window_total_events=int(snapshot["window_total_events"]),
                    window_oom_events=int(snapshot["window_oom_events"]),
                    window_oom_rate=float(snapshot["window_oom_rate"]),
                    unprocessed_events=int(snapshot["unprocessed_events"]),
                    last_trained_at=last_trained_raw,
                )

        snapshot = self.store.telemetry_trigger_snapshot(
            success_since=last_trained_at,
            window_start=now - timedelta(days=self.policy.oom_window_days),
        )
        success_count = int(snapshot["success_count_since"])
        total_events = int(snapshot["window_total_events"])
        oom_events = int(snapshot["window_oom_events"])
        oom_rate = float(snapshot["window_oom_rate"])
        backlog = int(snapshot["unprocessed_events"])

        if success_count >= self.policy.volume_threshold:
            return RetrainingDecision(
                should_retrain=True,
                reason=f"volume_threshold:{success_count}",
                success_count_since=success_count,
                window_total_events=total_events,
                window_oom_events=oom_events,
                window_oom_rate=oom_rate,
                unprocessed_events=backlog,
                last_trained_at=last_trained_raw,
            )
        if total_events >= self.policy.min_events_for_oom_rate and oom_rate > self.policy.oom_rate_threshold:
            return RetrainingDecision(
                should_retrain=True,
                reason=f"oom_rate_spike:{oom_rate:.4f}",
                success_count_since=success_count,
                window_total_events=total_events,
                window_oom_events=oom_events,
                window_oom_rate=oom_rate,
                unprocessed_events=backlog,
                last_trained_at=last_trained_raw,
            )
        return RetrainingDecision(
            should_retrain=False,
            reason="thresholds_not_met",
            success_count_since=success_count,
            window_total_events=total_events,
            window_oom_events=oom_events,
            window_oom_rate=oom_rate,
            unprocessed_events=backlog,
            last_trained_at=last_trained_raw,
        )

    def trigger_if_needed(
        self,
        *,
        dataset: str,
        run_training: bool = False,
    ) -> dict[str, Any]:
        decision = self.evaluate()
        if not decision.should_retrain:
            return {"status": "skipped", "decision": decision.to_dict()}
        if not run_training:
            return {"status": "eligible", "decision": decision.to_dict()}
        result = self.run_retraining_pipeline(dataset=dataset, trigger_reason=decision.reason)
        return {"status": "executed", "decision": decision.to_dict(), "result": result}

    def run_retraining_pipeline(self, *, dataset: str, trigger_reason: str) -> dict[str, Any]:
        candidate_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        python = sys.executable

        train_reranker = subprocess.run(
            [
                python,
                "scripts/train_ml_reranker.py",
                "--dataset",
                dataset,
            ],
            capture_output=True,
            text=True,
        )
        train_predictor = subprocess.run(
            [
                python,
                "scripts/train_hp_predictor.py",
                "--dataset",
                dataset,
            ],
            capture_output=True,
            text=True,
        )
        if train_reranker.returncode != 0 or train_predictor.returncode != 0:
            return {
                "status": "failed",
                "candidate_version": candidate_version,
                "stage": "training",
                "reranker_rc": train_reranker.returncode,
                "predictor_rc": train_predictor.returncode,
                "reranker_stderr": train_reranker.stderr[-2000:],
                "predictor_stderr": train_predictor.stderr[-2000:],
            }

        eval_run = subprocess.run(
            [
                python,
                "scripts/evaluate_candidate.py",
                "--candidate-version",
                candidate_version,
                "--dataset",
                dataset,
                "--auto-promote-to-staging",
            ],
            capture_output=True,
            text=True,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        state = self._load_state()
        state["last_trigger_reason"] = trigger_reason
        state["last_candidate_version"] = candidate_version
        state["last_eval_return_code"] = eval_run.returncode
        if eval_run.returncode == 0:
            state["last_trained_at"] = now_iso
        self._save_state(state)

        if eval_run.returncode != 0:
            self.governance.promote(
                model_name=DEFAULT_MODEL_NAME,
                version=candidate_version,
                to_state="REJECTED",
                actor="automated_pipeline",
                reason=f"evaluation_failed:{trigger_reason}",
                automated=True,
            )
            return {
                "status": "rejected",
                "candidate_version": candidate_version,
                "stage": "evaluation",
                "evaluation_rc": eval_run.returncode,
                "evaluation_stdout": eval_run.stdout[-4000:],
                "evaluation_stderr": eval_run.stderr[-4000:],
            }

        return {
            "status": "staged",
            "candidate_version": candidate_version,
            "evaluation_rc": eval_run.returncode,
            "evaluation_stdout": eval_run.stdout[-2000:],
        }

