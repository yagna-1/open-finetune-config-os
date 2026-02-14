from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import EngineStore

try:  # pragma: no cover - optional dependency
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover
    mlflow = None
    MlflowClient = None


DEFAULT_MODEL_NAME = "finetune_config_reranker"

VALID_TRANSITIONS: dict[str, set[str]] = {
    "TRAINING": {"CANDIDATE"},
    "CANDIDATE": {"STAGING", "REJECTED"},
    "REJECTED": {"CANDIDATE"},
    "STAGING": {"CANARY", "REJECTED"},
    "CANARY": {"PRODUCTION", "STAGING"},
    "PRODUCTION": {"DEPRECATED"},
    "DEPRECATED": {"PRODUCTION"},
    "UNKNOWN": {"CANDIDATE", "STAGING", "PRODUCTION", "REJECTED"},
}

MLFLOW_STAGE_MAP = {
    "CANDIDATE": "None",
    "STAGING": "Staging",
    "CANARY": "Staging",
    "PRODUCTION": "Production",
    "DEPRECATED": "Archived",
    "REJECTED": "Archived",
}


class GovernanceError(ValueError):
    pass


@dataclass(slots=True)
class PromotionResult:
    model_name: str
    version: str
    from_state: str
    to_state: str
    actor: str
    automated: bool
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "actor": self.actor,
            "automated": self.automated,
            "timestamp": self.timestamp,
        }


class ModelGovernanceService:
    def __init__(
        self,
        store: EngineStore,
        *,
        registry_path: str | Path = "artifacts/model_registry.json",
        tracking_uri: str | None = None,
    ) -> None:
        self.store = store
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = (tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")).strip() or None
        self.mlflow_client = None
        if mlflow is not None and MlflowClient is not None and self.tracking_uri:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                self.mlflow_client = MlflowClient(tracking_uri=self.tracking_uri)
            except Exception:
                self.mlflow_client = None

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"models": {}}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            return {"models": {}}

    def _save_registry(self, payload: dict) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _model_bucket(self, payload: dict, model_name: str) -> dict:
        models = payload.setdefault("models", {})
        return models.setdefault(model_name, {"versions": {}, "production": None, "canary": None})

    def current_state(self, model_name: str, version: str) -> str:
        payload = self._load_registry()
        bucket = self._model_bucket(payload, model_name)
        version_info = dict(bucket.get("versions", {})).get(str(version))
        if not version_info:
            return "UNKNOWN"
        return str(version_info.get("state") or "UNKNOWN")

    def list_versions(self, model_name: str) -> dict:
        payload = self._load_registry()
        bucket = self._model_bucket(payload, model_name)
        return {
            "model_name": model_name,
            "production": bucket.get("production"),
            "canary": bucket.get("canary"),
            "versions": dict(bucket.get("versions", {})),
        }

    def _validate_transition(self, from_state: str, to_state: str) -> None:
        allowed = VALID_TRANSITIONS.get(from_state, set())
        if to_state not in allowed:
            raise GovernanceError(f"invalid_transition:{from_state}->{to_state}")

    def promote(
        self,
        *,
        model_name: str,
        version: str,
        to_state: str,
        actor: str,
        reason: str = "",
        automated: bool = False,
        gate_results: dict | None = None,
    ) -> PromotionResult:
        normalized_to_state = to_state.strip().upper()
        if normalized_to_state not in MLFLOW_STAGE_MAP and normalized_to_state not in {"TRAINING", "UNKNOWN"}:
            raise GovernanceError(f"unsupported_state:{to_state}")

        payload = self._load_registry()
        bucket = self._model_bucket(payload, model_name)
        versions = bucket.setdefault("versions", {})
        previous = dict(versions.get(str(version), {}))
        from_state = str(previous.get("state") or "UNKNOWN").upper()
        self._validate_transition(from_state=from_state, to_state=normalized_to_state)

        now = datetime.now(timezone.utc).isoformat()
        versions[str(version)] = {
            "state": normalized_to_state,
            "updated_at": now,
            "actor": actor,
            "reason": reason,
            "automated": bool(automated),
        }

        if normalized_to_state == "PRODUCTION":
            old_prod = bucket.get("production")
            if old_prod and str(old_prod) != str(version):
                old_info = dict(versions.get(str(old_prod), {}))
                old_info["state"] = "DEPRECATED"
                old_info["updated_at"] = now
                old_info["actor"] = actor
                old_info["reason"] = "auto_deprecated_on_new_production"
                versions[str(old_prod)] = old_info
                self._record_event(
                    model_name=model_name,
                    version=str(old_prod),
                    from_state="PRODUCTION",
                    to_state="DEPRECATED",
                    actor=actor,
                    reason="auto_deprecated_on_new_production",
                    automated=automated,
                    gate_results=None,
                )
            bucket["production"] = str(version)
        elif normalized_to_state == "CANARY":
            bucket["canary"] = str(version)
        elif bucket.get("canary") == str(version):
            bucket["canary"] = None

        self._save_registry(payload)
        self._transition_mlflow(
            model_name=model_name,
            version=str(version),
            to_state=normalized_to_state,
            actor=actor,
            reason=reason,
        )

        self._record_event(
            model_name=model_name,
            version=str(version),
            from_state=from_state,
            to_state=normalized_to_state,
            actor=actor,
            reason=reason,
            automated=automated,
            gate_results=gate_results,
        )
        if normalized_to_state == "PRODUCTION":
            self._write_production_metrics_baseline(gate_results=gate_results)
        return PromotionResult(
            model_name=model_name,
            version=str(version),
            from_state=from_state,
            to_state=normalized_to_state,
            actor=actor,
            automated=automated,
            timestamp=now,
        )

    def promote_to_staging(
        self,
        *,
        version: str,
        model_name: str = DEFAULT_MODEL_NAME,
        actor: str = "automated_pipeline",
        reason: str = "all_eval_gates_passed",
        gate_results: dict | None = None,
    ) -> PromotionResult:
        return self.promote(
            model_name=model_name,
            version=version,
            to_state="STAGING",
            actor=actor,
            reason=reason,
            automated=True,
            gate_results=gate_results,
        )

    def rollback(
        self,
        *,
        model_name: str,
        actor: str,
        reason: str,
        to_version: str | None = None,
    ) -> dict[str, Any]:
        bucket = self.list_versions(model_name)
        versions = dict(bucket.get("versions", {}))
        current_prod = bucket.get("production")
        target = to_version
        if target is None:
            deprecated = [
                (ver, info)
                for ver, info in versions.items()
                if str(info.get("state")).upper() == "DEPRECATED"
            ]
            deprecated.sort(key=lambda item: int(item[0]) if str(item[0]).isdigit() else -1, reverse=True)
            if not deprecated:
                raise GovernanceError("rollback_target_not_found")
            target = str(deprecated[0][0])

        if current_prod and str(current_prod) != str(target):
            self.promote(
                model_name=model_name,
                version=str(current_prod),
                to_state="DEPRECATED",
                actor=actor,
                reason=f"rollback_triggered:{reason}",
                automated=False,
            )

        result = self.promote(
            model_name=model_name,
            version=str(target),
            to_state="PRODUCTION",
            actor=actor,
            reason=reason,
            automated=False,
        )
        return {
            "rolled_back_to": str(target),
            "previous_production": str(current_prod) if current_prod else None,
            "result": result.to_dict(),
        }

    def _record_event(
        self,
        *,
        model_name: str,
        version: str,
        from_state: str,
        to_state: str,
        actor: str,
        reason: str,
        automated: bool,
        gate_results: dict | None,
    ) -> None:
        self.store.record_governance_event(
            model_name=model_name,
            version=version,
            from_state=from_state,
            to_state=to_state,
            actor=actor,
            reason=reason,
            automated=automated,
            gate_results=gate_results,
        )

    def _transition_mlflow(
        self,
        *,
        model_name: str,
        version: str,
        to_state: str,
        actor: str,
        reason: str,
    ) -> None:
        if self.mlflow_client is None:
            return
        stage = MLFLOW_STAGE_MAP.get(to_state)
        try:
            if stage is not None:
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage=stage,
                )
            self.mlflow_client.set_model_version_tag(model_name, version, "current_state", to_state)
            self.mlflow_client.set_model_version_tag(model_name, version, "promoted_by", actor)
            self.mlflow_client.set_model_version_tag(model_name, version, "promotion_reason", reason or "")
            self.mlflow_client.set_model_version_tag(
                model_name,
                version,
                "promoted_at",
                datetime.now(timezone.utc).isoformat(),
            )
            if to_state == "CANARY":
                self.mlflow_client.set_model_version_tag(model_name, version, "traffic_weight", "0.05")
        except Exception:
            # Governance must not crash if MLflow is unavailable.
            return

    def _write_production_metrics_baseline(self, *, gate_results: dict | None) -> None:
        if not gate_results:
            return
        report_path_value = gate_results.get("report_path")
        if not report_path_value:
            return
        report_path = Path(str(report_path_value))
        if not report_path.exists():
            return
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            return
        baseline = {
            "overall_accuracy": report_payload.get("overall_accuracy", 0.0),
            "per_category_acc": report_payload.get("per_category_acc", {}),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source_report": str(report_path),
        }
        output = Path("artifacts/eval_reports/production_metrics.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(baseline, indent=2, ensure_ascii=True), encoding="utf-8")
