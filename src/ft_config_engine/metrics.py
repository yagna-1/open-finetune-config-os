from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - optional dependency
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = None
    Gauge = None
    Histogram = None
    generate_latest = None


@dataclass
class MetricsRegistry:
    enabled: bool

    def __post_init__(self) -> None:
        self.recommendations_total = (
            Counter(
                "recommendations_total",
                "Total recommendation requests served",
                labelnames=["task_type", "adapter_type", "confidence_level", "strategy", "model_slot"],
            )
            if self.enabled
            else None
        )
        self.telemetry_events_total = (
            Counter(
                "telemetry_events_total",
                "Telemetry events received by outcome type",
                labelnames=["event_type", "outcome"],
            )
            if self.enabled
            else None
        )
        self.oom_violations_total = (
            Counter(
                "oom_violations_total",
                "Telemetry events reporting OOM outcome",
                labelnames=["gpu_type", "adapter_type", "model_size_bucket"],
            )
            if self.enabled
            else None
        )
        self.model_governance_transitions_total = (
            Counter(
                "model_governance_transitions_total",
                "Model lifecycle state transitions",
                labelnames=["model_name", "from_state", "to_state", "automated"],
            )
            if self.enabled
            else None
        )
        self.recommendation_latency_seconds = (
            Histogram(
                "recommendation_latency_seconds",
                "End-to-end recommendation latency",
                labelnames=["strategy"],
                buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
            )
            if self.enabled
            else None
        )
        self.confidence_scores = (
            Histogram(
                "confidence_scores",
                "Distribution of recommendation confidence scores",
                labelnames=["task_type", "fallback_level"],
                buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            )
            if self.enabled
            else None
        )
        self.telemetry_events_pending = (
            Gauge(
                "telemetry_events_pending",
                "Unprocessed telemetry events in queue",
            )
            if self.enabled
            else None
        )
        self.hours_since_last_retraining = (
            Gauge(
                "hours_since_last_retraining",
                "Hours elapsed since last successful model retraining",
            )
            if self.enabled
            else None
        )

    def latency_timer(self, strategy: str):
        if not self.enabled or self.recommendation_latency_seconds is None:
            class _NoopTimer:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

            return _NoopTimer()
        return self.recommendation_latency_seconds.labels(strategy=strategy).time()

    def inc_recommendation(
        self,
        *,
        task_type: str,
        adapter_type: str,
        confidence_level: str,
        strategy: str,
        model_slot: str,
    ) -> None:
        if not self.enabled or self.recommendations_total is None:
            return
        self.recommendations_total.labels(
            task_type=task_type,
            adapter_type=adapter_type,
            confidence_level=confidence_level,
            strategy=strategy,
            model_slot=model_slot,
        ).inc()

    def observe_confidence(self, *, task_type: str, fallback_level: str, confidence_score: float) -> None:
        if not self.enabled or self.confidence_scores is None:
            return
        self.confidence_scores.labels(task_type=task_type, fallback_level=fallback_level).observe(confidence_score)

    def inc_telemetry(self, *, event_type: str, outcome: str) -> None:
        if not self.enabled or self.telemetry_events_total is None:
            return
        self.telemetry_events_total.labels(event_type=event_type, outcome=outcome).inc()

    def inc_oom_violation(self, *, gpu_type: str, adapter_type: str, model_size_bucket: str) -> None:
        if not self.enabled or self.oom_violations_total is None:
            return
        self.oom_violations_total.labels(
            gpu_type=gpu_type or "unknown",
            adapter_type=adapter_type or "unknown",
            model_size_bucket=model_size_bucket or "unknown",
        ).inc()

    def inc_governance_transition(
        self,
        *,
        model_name: str,
        from_state: str,
        to_state: str,
        automated: bool,
    ) -> None:
        if not self.enabled or self.model_governance_transitions_total is None:
            return
        self.model_governance_transitions_total.labels(
            model_name=model_name,
            from_state=from_state,
            to_state=to_state,
            automated=str(bool(automated)).lower(),
        ).inc()

    def set_telemetry_backlog(self, pending: int) -> None:
        if not self.enabled or self.telemetry_events_pending is None:
            return
        self.telemetry_events_pending.set(max(0, int(pending)))

    def set_retrain_lag_hours(self, hours: float) -> None:
        if not self.enabled or self.hours_since_last_retraining is None:
            return
        self.hours_since_last_retraining.set(max(0.0, float(hours)))

    def export_payload(self) -> bytes:
        if not self.enabled or generate_latest is None:
            return b""
        return generate_latest()


def create_metrics_registry() -> MetricsRegistry:
    return MetricsRegistry(enabled=Counter is not None and Gauge is not None and Histogram is not None)
