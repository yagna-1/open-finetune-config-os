from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column, sessionmaker

from .models import NormalizedConfig, StatisticalProfile

Base = declarative_base()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class NormalizedConfigRow(Base):
    __tablename__ = "normalized_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    dataset_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    model_size_bucket: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    adapter_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    effective_batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    max_seq_length: Mapped[int] = mapped_column(Integer, nullable=False)
    precision: Mapped[str] = mapped_column(String(32), nullable=False)
    optimizer: Mapped[str] = mapped_column(String(128), nullable=False)
    scheduler: Mapped[str] = mapped_column(String(128), nullable=False)
    lora_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_type: Mapped[str] = mapped_column(String(64), nullable=False)
    gpu_memory_gb: Mapped[float] = mapped_column(Float, nullable=False)
    num_gpus: Mapped[int] = mapped_column(Integer, nullable=False)
    performance_metric_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    performance_metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class StatisticalProfileRow(Base):
    __tablename__ = "statistical_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_key: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model_size_bucket: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    adapter_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    median_learning_rate: Mapped[float] = mapped_column(Float, nullable=False)
    learning_rate_q1: Mapped[float] = mapped_column(Float, nullable=False)
    learning_rate_q3: Mapped[float] = mapped_column(Float, nullable=False)
    learning_rate_iqr: Mapped[float] = mapped_column(Float, nullable=False)
    median_effective_batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    median_lora_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    typical_optimizer: Mapped[str] = mapped_column(String(128), nullable=False)
    typical_precision: Mapped[str] = mapped_column(String(32), nullable=False)
    median_seq_length: Mapped[int] = mapped_column(Integer, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class RecommendationEventRow(Base):
    __tablename__ = "recommendation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    selected_gpu: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    estimated_vram_gb_per_gpu: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_training_time_hours: Mapped[float] = mapped_column(Float, nullable=False)
    request_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    result_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class RecommendationFeedbackRow(Base):
    __tablename__ = "recommendation_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("recommendation_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class TelemetryRunStartRow(Base):
    __tablename__ = "telemetry_run_start"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    recommendation_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("recommendation_events.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    schema_version: Mapped[str] = mapped_column(String(8), nullable=False, default="v1")
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    model_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    adapter_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gpu_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    actual_lr: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_batch_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_gradient_accum: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_lora_r: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_epochs: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_max_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    was_config_modified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    recommendation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    fallback_level: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class TelemetryRunCompleteRow(Base):
    __tablename__ = "telemetry_run_complete"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    recommendation_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("recommendation_events.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    run_start_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("telemetry_run_start.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    final_train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_eval_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    primary_metric_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    primary_metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    failure_mode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    wall_clock_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_vram_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    user_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    user_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class ModelGovernanceAuditRow(Base):
    __tablename__ = "model_governance_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    from_state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    to_state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    gate_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    automated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


_PII_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
)


class EngineStore:
    def __init__(self, database_url: str) -> None:
        self.database_url = normalize_database_url(database_url)
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine = create_engine(
            self.database_url,
            echo=False,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

    def create_tables(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def sync_reference_data(
        self,
        configs: list[NormalizedConfig],
        profiles: dict[tuple[str, str, str], StatisticalProfile],
    ) -> dict[str, int]:
        inserted_configs = 0
        updated_configs = 0
        inserted_profiles = 0
        updated_profiles = 0

        with self.session_scope() as session:
            config_by_record = {
                row.record_id: row
                for row in session.execute(select(NormalizedConfigRow)).scalars().all()
            }
            for cfg in configs:
                payload = cfg.to_dict()
                existing = config_by_record.get(cfg.record_id)
                if existing is None:
                    row = NormalizedConfigRow(
                        record_id=cfg.record_id,
                        task_type=cfg.task_type,
                        dataset_name=cfg.dataset_name,
                        model_name=cfg.model_name,
                        model_size_bucket=cfg.model_size_bucket,
                        adapter_type=cfg.adapter_type,
                        learning_rate=cfg.learning_rate,
                        effective_batch_size=cfg.effective_batch_size,
                        max_seq_length=cfg.max_seq_length,
                        precision=cfg.precision,
                        optimizer=cfg.optimizer,
                        scheduler=cfg.scheduler,
                        lora_rank=cfg.lora_rank,
                        gpu_type=cfg.gpu_type,
                        gpu_memory_gb=cfg.gpu_memory_gb,
                        num_gpus=cfg.num_gpus,
                        performance_metric_name=cfg.performance_metric_name,
                        performance_metric_value=cfg.performance_metric_value,
                        validation_loss=cfg.validation_loss,
                        payload=payload,
                    )
                    session.add(row)
                    config_by_record[cfg.record_id] = row
                    inserted_configs += 1
                    continue

                existing.task_type = cfg.task_type
                existing.dataset_name = cfg.dataset_name
                existing.model_name = cfg.model_name
                existing.model_size_bucket = cfg.model_size_bucket
                existing.adapter_type = cfg.adapter_type
                existing.learning_rate = cfg.learning_rate
                existing.effective_batch_size = cfg.effective_batch_size
                existing.max_seq_length = cfg.max_seq_length
                existing.precision = cfg.precision
                existing.optimizer = cfg.optimizer
                existing.scheduler = cfg.scheduler
                existing.lora_rank = cfg.lora_rank
                existing.gpu_type = cfg.gpu_type
                existing.gpu_memory_gb = cfg.gpu_memory_gb
                existing.num_gpus = cfg.num_gpus
                existing.performance_metric_name = cfg.performance_metric_name
                existing.performance_metric_value = cfg.performance_metric_value
                existing.validation_loss = cfg.validation_loss
                existing.payload = payload
                existing.updated_at = utc_now()
                updated_configs += 1

            profile_by_key = {
                row.profile_key: row
                for row in session.execute(select(StatisticalProfileRow)).scalars().all()
            }
            for key, profile in profiles.items():
                profile_key = "|".join(key)
                payload = profile.to_dict()
                existing = profile_by_key.get(profile_key)
                if existing is None:
                    row = StatisticalProfileRow(
                        profile_key=profile_key,
                        task_type=profile.task_type,
                        model_size_bucket=profile.model_size_bucket,
                        adapter_type=profile.adapter_type,
                        sample_size=profile.sample_size,
                        median_learning_rate=profile.median_learning_rate,
                        learning_rate_q1=profile.learning_rate_q1,
                        learning_rate_q3=profile.learning_rate_q3,
                        learning_rate_iqr=profile.learning_rate_iqr,
                        median_effective_batch_size=profile.median_effective_batch_size,
                        median_lora_rank=profile.median_lora_rank,
                        typical_optimizer=profile.typical_optimizer,
                        typical_precision=profile.typical_precision,
                        median_seq_length=profile.median_seq_length,
                        payload=payload,
                    )
                    session.add(row)
                    profile_by_key[profile_key] = row
                    inserted_profiles += 1
                    continue

                existing.task_type = profile.task_type
                existing.model_size_bucket = profile.model_size_bucket
                existing.adapter_type = profile.adapter_type
                existing.sample_size = profile.sample_size
                existing.median_learning_rate = profile.median_learning_rate
                existing.learning_rate_q1 = profile.learning_rate_q1
                existing.learning_rate_q3 = profile.learning_rate_q3
                existing.learning_rate_iqr = profile.learning_rate_iqr
                existing.median_effective_batch_size = profile.median_effective_batch_size
                existing.median_lora_rank = profile.median_lora_rank
                existing.typical_optimizer = profile.typical_optimizer
                existing.typical_precision = profile.typical_precision
                existing.median_seq_length = profile.median_seq_length
                existing.payload = payload
                existing.updated_at = utc_now()
                updated_profiles += 1

        return {
            "inserted_configs": inserted_configs,
            "updated_configs": updated_configs,
            "inserted_profiles": inserted_profiles,
            "updated_profiles": updated_profiles,
        }

    def log_recommendation(self, request_payload: dict, result_payload: dict) -> int:
        strategy = (
            result_payload.get("recommendation_basis", {}).get("strategy")
            or request_payload.get("strategy")
            or "unknown"
        )
        selected_gpu = str(result_payload.get("selected_gpu") or "unknown")
        est_vram = float(result_payload.get("estimated_vram_gb_per_gpu") or 0.0)
        est_hours = float(result_payload.get("estimated_training_time_hours") or 0.0)

        with self.session_scope() as session:
            row = RecommendationEventRow(
                strategy=strategy,
                selected_gpu=selected_gpu,
                estimated_vram_gb_per_gpu=est_vram,
                estimated_training_time_hours=est_hours,
                request_payload=request_payload,
                result_payload=result_payload,
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def list_profiles(
        self,
        task_type: str | None = None,
        model_size_bucket: str | None = None,
        adapter_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        with self.session_scope() as session:
            stmt = select(StatisticalProfileRow).order_by(StatisticalProfileRow.sample_size.desc())
            if task_type:
                stmt = stmt.where(StatisticalProfileRow.task_type == task_type)
            if model_size_bucket:
                stmt = stmt.where(StatisticalProfileRow.model_size_bucket == model_size_bucket)
            if adapter_type:
                stmt = stmt.where(StatisticalProfileRow.adapter_type == adapter_type)
            rows = session.execute(stmt.limit(max(1, min(limit, 1000)))).scalars().all()
            return [row.payload for row in rows]

    def list_recommendations(self, limit: int = 50) -> list[dict]:
        with self.session_scope() as session:
            rows = session.execute(
                select(RecommendationEventRow)
                .order_by(RecommendationEventRow.created_at.desc())
                .limit(max(1, min(limit, 500)))
            ).scalars().all()
            return [
                {
                    "id": row.id,
                    "strategy": row.strategy,
                    "selected_gpu": row.selected_gpu,
                    "estimated_vram_gb_per_gpu": row.estimated_vram_gb_per_gpu,
                    "estimated_training_time_hours": row.estimated_training_time_hours,
                    "created_at": row.created_at.isoformat(),
                    "request_payload": row.request_payload,
                    "result_payload": row.result_payload,
                }
                for row in rows
            ]

    def submit_feedback(
        self,
        recommendation_event_id: int,
        rating: int | None = None,
        success: bool | None = None,
        notes: str | None = None,
    ) -> dict:
        with self.session_scope() as session:
            event = session.get(RecommendationEventRow, recommendation_event_id)
            if event is None:
                raise ValueError("recommendation_event_not_found")

            feedback = RecommendationFeedbackRow(
                recommendation_event_id=recommendation_event_id,
                rating=rating,
                success=success,
                notes=notes,
            )
            session.add(feedback)
            session.flush()
            return {
                "id": int(feedback.id),
                "recommendation_event_id": recommendation_event_id,
                "rating": rating,
                "success": success,
                "notes": notes,
                "created_at": feedback.created_at.isoformat(),
            }

    def record_telemetry_run_start(self, payload: Mapping[str, object]) -> dict:
        recommendation_event_id = self._optional_int(payload.get("recommendation_event_id"))
        if recommendation_event_id is not None:
            self._ensure_recommendation_event_exists(recommendation_event_id)

        dataset_name = str(payload.get("dataset_name") or "").strip()
        dataset_hash = hashlib.blake2s(dataset_name.encode("utf-8"), digest_size=16).hexdigest()

        actual_lr = self._optional_float(payload.get("actual_lr"))
        actual_batch_size = self._optional_int(payload.get("actual_batch_size"))
        actual_gradient_accum = self._optional_int(payload.get("actual_gradient_accum"))
        actual_lora_r = self._optional_int(payload.get("actual_lora_r"))
        actual_epochs = self._optional_float(payload.get("actual_epochs"))
        actual_max_steps = self._optional_int(payload.get("actual_max_steps"))
        recommendation_confidence = self._optional_float(payload.get("recommendation_confidence"))
        wall_dataset_size = self._optional_int(payload.get("dataset_size"))

        self._check_numeric_range(actual_lr, field="actual_lr", low=1e-9, high=1.0)
        self._check_numeric_range(
            recommendation_confidence,
            field="recommendation_confidence",
            low=0.0,
            high=1.0,
        )
        self._check_numeric_range(
            float(actual_batch_size) if actual_batch_size is not None else None,
            field="actual_batch_size",
            low=1.0,
            high=4096.0,
        )
        self._check_numeric_range(
            float(actual_gradient_accum) if actual_gradient_accum is not None else None,
            field="actual_gradient_accum",
            low=1.0,
            high=512.0,
        )
        self._check_numeric_range(
            float(actual_lora_r) if actual_lora_r is not None else None,
            field="actual_lora_r",
            low=1.0,
            high=512.0,
        )

        with self.session_scope() as session:
            row = TelemetryRunStartRow(
                recommendation_event_id=recommendation_event_id,
                schema_version=str(payload.get("schema_version") or "v1")[:8],
                model_id=str(payload.get("model_id") or "")[:256],
                task_type=str(payload.get("task_type") or "")[:64],
                adapter_type=str(payload.get("adapter_type") or "")[:16],
                dataset_hash=dataset_hash,
                dataset_size=wall_dataset_size,
                gpu_type=(str(payload.get("gpu_type"))[:64] if payload.get("gpu_type") else None),
                actual_lr=actual_lr,
                actual_batch_size=actual_batch_size,
                actual_gradient_accum=actual_gradient_accum,
                actual_lora_r=actual_lora_r,
                actual_epochs=actual_epochs,
                actual_max_steps=actual_max_steps,
                was_config_modified=bool(payload.get("was_config_modified") or False),
                recommendation_confidence=recommendation_confidence,
                fallback_level=(str(payload.get("fallback_level"))[:32] if payload.get("fallback_level") else None),
                payload=dict(payload),
            )
            session.add(row)
            session.flush()
            return {
                "id": row.id,
                "recommendation_event_id": row.recommendation_event_id,
                "received_at": row.received_at.isoformat(),
                "dataset_hash": row.dataset_hash,
            }

    def record_telemetry_run_complete(self, payload: Mapping[str, object]) -> dict:
        recommendation_event_id = self._optional_int(payload.get("recommendation_event_id"))
        run_start_id = (str(payload.get("run_start_id"))[:36] if payload.get("run_start_id") else None)
        if recommendation_event_id is not None:
            self._ensure_recommendation_event_exists(recommendation_event_id)
        if run_start_id is not None:
            self._ensure_run_start_exists(run_start_id)

        final_train_loss = self._optional_float(payload.get("final_train_loss"))
        final_eval_loss = self._optional_float(payload.get("final_eval_loss"))
        primary_metric_value = self._optional_float(payload.get("primary_metric_value"))
        failure_step = self._optional_int(payload.get("failure_step"))
        wall_clock_minutes = self._optional_float(payload.get("wall_clock_minutes"))
        peak_vram_gb = self._optional_float(payload.get("peak_vram_gb"))
        tokens_per_second = self._optional_float(payload.get("tokens_per_second"))
        estimated_cost_usd = self._optional_float(payload.get("estimated_cost_usd"))
        user_rating = self._optional_int(payload.get("user_rating"))

        self._check_numeric_range(wall_clock_minutes, field="wall_clock_minutes", low=0.0, high=100_000.0)
        self._check_numeric_range(peak_vram_gb, field="peak_vram_gb", low=0.0, high=500.0)
        self._check_numeric_range(tokens_per_second, field="tokens_per_second", low=0.0, high=10_000_000.0)
        self._check_numeric_range(estimated_cost_usd, field="estimated_cost_usd", low=0.0, high=100_000.0)
        self._check_numeric_range(
            float(user_rating) if user_rating is not None else None,
            field="user_rating",
            low=1.0,
            high=5.0,
        )

        clean_failure_message = self._redact_pii(str(payload.get("failure_message") or "")) or None
        clean_user_note = self._redact_pii(str(payload.get("user_note") or "")) or None

        with self.session_scope() as session:
            row = TelemetryRunCompleteRow(
                recommendation_event_id=recommendation_event_id,
                run_start_id=run_start_id,
                outcome=str(payload.get("outcome") or "unknown")[:32],
                final_train_loss=final_train_loss,
                final_eval_loss=final_eval_loss,
                primary_metric_name=(
                    str(payload.get("primary_metric_name"))[:64] if payload.get("primary_metric_name") else None
                ),
                primary_metric_value=primary_metric_value,
                failure_mode=(str(payload.get("failure_mode"))[:64] if payload.get("failure_mode") else None),
                failure_step=failure_step,
                failure_message=clean_failure_message,
                wall_clock_minutes=wall_clock_minutes,
                peak_vram_gb=peak_vram_gb,
                tokens_per_second=tokens_per_second,
                estimated_cost_usd=estimated_cost_usd,
                user_rating=user_rating,
                user_note=clean_user_note,
                processed=False,
                processed_at=None,
                payload=dict(payload),
            )
            session.add(row)
            session.flush()
            return {
                "id": row.id,
                "recommendation_event_id": row.recommendation_event_id,
                "run_start_id": row.run_start_id,
                "received_at": row.received_at.isoformat(),
                "processed": row.processed,
            }

    def list_telemetry_runs(self, limit: int = 100) -> dict:
        effective_limit = max(1, min(limit, 1000))
        with self.session_scope() as session:
            starts = session.execute(
                select(TelemetryRunStartRow)
                .order_by(TelemetryRunStartRow.received_at.desc())
                .limit(effective_limit)
            ).scalars().all()
            completes = session.execute(
                select(TelemetryRunCompleteRow)
                .order_by(TelemetryRunCompleteRow.received_at.desc())
                .limit(effective_limit)
            ).scalars().all()
        return {
            "run_start": [
                {
                    "id": row.id,
                    "recommendation_event_id": row.recommendation_event_id,
                    "schema_version": row.schema_version,
                    "model_id": row.model_id,
                    "task_type": row.task_type,
                    "adapter_type": row.adapter_type,
                    "dataset_hash": row.dataset_hash,
                    "dataset_size": row.dataset_size,
                    "gpu_type": row.gpu_type,
                    "received_at": row.received_at.isoformat(),
                }
                for row in starts
            ],
            "run_complete": [
                {
                    "id": row.id,
                    "recommendation_event_id": row.recommendation_event_id,
                    "run_start_id": row.run_start_id,
                    "outcome": row.outcome,
                    "failure_mode": row.failure_mode,
                    "wall_clock_minutes": row.wall_clock_minutes,
                    "peak_vram_gb": row.peak_vram_gb,
                    "estimated_cost_usd": row.estimated_cost_usd,
                    "received_at": row.received_at.isoformat(),
                    "processed": row.processed,
                }
                for row in completes
            ],
        }

    def telemetry_trigger_snapshot(
        self,
        *,
        success_since: datetime | None = None,
        window_start: datetime | None = None,
    ) -> dict[str, float]:
        with self.session_scope() as session:
            success_stmt = select(TelemetryRunCompleteRow).where(TelemetryRunCompleteRow.outcome == "success")
            if success_since is not None:
                success_stmt = success_stmt.where(TelemetryRunCompleteRow.received_at > success_since)
            success_count = float(len(session.execute(success_stmt).scalars().all()))

            window_stmt = select(TelemetryRunCompleteRow)
            if window_start is not None:
                window_stmt = window_stmt.where(TelemetryRunCompleteRow.received_at > window_start)
            window_rows = session.execute(window_stmt).scalars().all()
            total_events = float(len(window_rows))
            oom_events = float(sum(1 for row in window_rows if str(row.outcome).lower() == "oom"))

            unprocessed_count = float(
                len(
                    session.execute(
                        select(TelemetryRunCompleteRow).where(TelemetryRunCompleteRow.processed.is_(False))
                    ).scalars().all()
                )
            )

        return {
            "success_count_since": success_count,
            "window_total_events": total_events,
            "window_oom_events": oom_events,
            "window_oom_rate": (oom_events / total_events) if total_events > 0 else 0.0,
            "unprocessed_events": unprocessed_count,
        }

    def mark_telemetry_processed(self, event_ids: list[str]) -> int:
        if not event_ids:
            return 0
        with self.session_scope() as session:
            rows = session.execute(
                select(TelemetryRunCompleteRow).where(TelemetryRunCompleteRow.id.in_(event_ids))
            ).scalars().all()
            for row in rows:
                row.processed = True
                row.processed_at = utc_now()
            return len(rows)

    def record_governance_event(
        self,
        *,
        model_name: str,
        version: str,
        from_state: str,
        to_state: str,
        actor: str,
        reason: str | None = None,
        gate_results: dict | None = None,
        automated: bool = False,
    ) -> dict:
        with self.session_scope() as session:
            row = ModelGovernanceAuditRow(
                model_name=model_name[:128],
                version=version[:32],
                from_state=from_state[:32],
                to_state=to_state[:32],
                actor=actor[:128],
                reason=(reason[:2000] if reason else None),
                gate_results=gate_results,
                automated=automated,
            )
            session.add(row)
            session.flush()
            return {
                "id": int(row.id),
                "model_name": row.model_name,
                "version": row.version,
                "from_state": row.from_state,
                "to_state": row.to_state,
                "actor": row.actor,
                "timestamp": row.timestamp.isoformat(),
                "automated": row.automated,
            }

    def list_governance_events(
        self,
        *,
        model_name: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        with self.session_scope() as session:
            stmt = select(ModelGovernanceAuditRow).order_by(ModelGovernanceAuditRow.timestamp.desc())
            if model_name:
                stmt = stmt.where(ModelGovernanceAuditRow.model_name == model_name)
            rows = session.execute(stmt.limit(max(1, min(limit, 1000)))).scalars().all()
            return [
                {
                    "id": int(row.id),
                    "model_name": row.model_name,
                    "version": row.version,
                    "from_state": row.from_state,
                    "to_state": row.to_state,
                    "actor": row.actor,
                    "timestamp": row.timestamp.isoformat(),
                    "reason": row.reason,
                    "gate_results": row.gate_results or {},
                    "automated": row.automated,
                }
                for row in rows
            ]

    def canary_rollout_summary(self, *, lookback: int = 500) -> dict:
        with self.session_scope() as session:
            rows = session.execute(
                select(RecommendationEventRow)
                .order_by(RecommendationEventRow.created_at.desc())
                .limit(max(1, min(lookback, 10_000)))
            ).scalars().all()

        slot_aggregates: dict[str, dict[str, float]] = {}
        for row in rows:
            request_payload = dict(row.request_payload or {})
            result_payload = dict(row.result_payload or {})
            basis = dict(result_payload.get("recommendation_basis") or {})
            slot = str(request_payload.get("_model_slot") or basis.get("model_slot") or "production")
            aggregate = slot_aggregates.setdefault(
                slot,
                {
                    "requests": 0.0,
                    "avg_confidence": 0.0,
                    "avg_vram_ratio": 0.0,
                    "high_risk_count": 0.0,
                },
            )
            aggregate["requests"] += 1.0
            confidence = float(basis.get("confidence_score") or 0.0)
            aggregate["avg_confidence"] += confidence
            est_vram = float(result_payload.get("estimated_vram_gb_per_gpu") or 0.0)
            gpu_mem = float(result_payload.get("selected_gpu_memory_gb") or 0.0)
            ratio = est_vram / gpu_mem if gpu_mem > 0 else 0.0
            aggregate["avg_vram_ratio"] += ratio
            if ratio > 0.9:
                aggregate["high_risk_count"] += 1.0

        summary: dict[str, dict] = {}
        total_requests = sum(values["requests"] for values in slot_aggregates.values())
        for slot, values in slot_aggregates.items():
            requests = max(1.0, values["requests"])
            summary[slot] = {
                "requests": int(values["requests"]),
                "traffic_share": float(values["requests"] / total_requests) if total_requests > 0 else 0.0,
                "avg_confidence": float(values["avg_confidence"] / requests),
                "avg_vram_ratio": float(values["avg_vram_ratio"] / requests),
                "high_risk_rate": float(values["high_risk_count"] / requests),
            }
        return {
            "lookback": int(min(max(1, lookback), 10_000)),
            "total_requests": int(total_requests),
            "slots": summary,
        }

    @staticmethod
    def _redact_pii(value: str) -> str:
        text = value
        for pattern in _PII_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        return text[:500]

    @staticmethod
    def _check_numeric_range(
        value: float | None,
        *,
        field: str,
        low: float,
        high: float,
    ) -> None:
        if value is None:
            return
        if not (low <= value <= high):
            raise ValueError(f"{field}_out_of_range")

    def _ensure_recommendation_event_exists(self, recommendation_event_id: int) -> None:
        with self.session_scope() as session:
            if session.get(RecommendationEventRow, recommendation_event_id) is None:
                raise ValueError("recommendation_event_not_found")

    def _ensure_run_start_exists(self, run_start_id: str) -> None:
        with self.session_scope() as session:
            if session.get(TelemetryRunStartRow, run_start_id) is None:
                raise ValueError("telemetry_run_start_not_found")

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    def learned_strategy_for_context(
        self,
        task_type: str,
        adapter_type: str,
        model_size_bucket: str | None,
        *,
        min_feedback: int = 2,
        lookback: int = 500,
    ) -> dict:
        with self.session_scope() as session:
            recent_rows = session.execute(
                select(RecommendationEventRow)
                .order_by(RecommendationEventRow.created_at.desc())
                .limit(max(1, min(lookback, 5000)))
            ).scalars().all()

            filtered_events: list[RecommendationEventRow] = []
            for row in recent_rows:
                basis = dict(row.result_payload.get("recommendation_basis") or {})
                hparams = dict(row.result_payload.get("safe_hyperparameters") or {})

                row_task = str(
                    basis.get("resolved_task_type")
                    or row.request_payload.get("task_type")
                    or ""
                ).strip().lower()
                row_adapter = str(
                    hparams.get("adapter_type")
                    or row.request_payload.get("adapter_type")
                    or ""
                ).strip().lower()
                row_bucket = (
                    str(
                        basis.get("model_size_bucket")
                        or row.request_payload.get("model_size_bucket")
                        or ""
                    )
                    .strip()
                    .lower()
                )

                if row_task != task_type.strip().lower():
                    continue
                if row_adapter != adapter_type.strip().lower():
                    continue
                if model_size_bucket and row_bucket != model_size_bucket.strip().lower():
                    continue
                filtered_events.append(row)

            if not filtered_events:
                return {
                    "strategy": None,
                    "reason": "no-context-events",
                    "context": {
                        "task_type": task_type,
                        "adapter_type": adapter_type,
                        "model_size_bucket": model_size_bucket,
                    },
                    "feedback_samples": 0,
                    "strategy_scores": {},
                }

            event_ids = [row.id for row in filtered_events]
            feedback_rows = session.execute(
                select(RecommendationFeedbackRow)
                .where(RecommendationFeedbackRow.recommendation_event_id.in_(event_ids))
                .order_by(RecommendationFeedbackRow.created_at.desc())
            ).scalars().all()

            feedback_by_event: dict[int, RecommendationFeedbackRow] = {}
            for feedback in feedback_rows:
                if feedback.recommendation_event_id not in feedback_by_event:
                    feedback_by_event[feedback.recommendation_event_id] = feedback

            aggregates: dict[str, dict[str, float]] = {}
            total_feedback = 0
            for row in filtered_events:
                strategy = str(row.strategy or "deterministic")
                aggregate = aggregates.setdefault(
                    strategy,
                    {
                        "events": 0.0,
                        "feedback": 0.0,
                        "rating_sum": 0.0,
                        "rating_count": 0.0,
                        "success_sum": 0.0,
                        "success_count": 0.0,
                        "confidence_sum": 0.0,
                        "risk_sum": 0.0,
                    },
                )
                aggregate["events"] += 1.0

                basis = dict(row.result_payload.get("recommendation_basis") or {})
                confidence = float(basis.get("confidence_score") or 0.5)
                aggregate["confidence_sum"] += confidence

                est_vram = float(row.result_payload.get("estimated_vram_gb_per_gpu") or 0.0)
                gpu_mem = float(row.result_payload.get("selected_gpu_memory_gb") or 0.0)
                risk_ratio = est_vram / gpu_mem if gpu_mem > 0 else 0.0
                aggregate["risk_sum"] += max(0.0, min(2.0, risk_ratio))

                feedback = feedback_by_event.get(row.id)
                if feedback is None:
                    continue

                aggregate["feedback"] += 1.0
                total_feedback += 1
                if feedback.rating is not None:
                    aggregate["rating_sum"] += float(feedback.rating)
                    aggregate["rating_count"] += 1.0
                if feedback.success is not None:
                    aggregate["success_sum"] += 1.0 if feedback.success else 0.0
                    aggregate["success_count"] += 1.0

            strategy_scores: dict[str, float] = {}
            for strategy, aggregate in aggregates.items():
                event_count = max(1.0, aggregate["events"])
                feedback_count = aggregate["feedback"]

                rating_score = (
                    (aggregate["rating_sum"] / aggregate["rating_count"]) / 5.0
                    if aggregate["rating_count"] > 0
                    else 0.55
                )
                success_score = (
                    aggregate["success_sum"] / aggregate["success_count"]
                    if aggregate["success_count"] > 0
                    else 0.55
                )
                confidence_score = aggregate["confidence_sum"] / event_count
                avg_risk = aggregate["risk_sum"] / event_count
                safety_score = max(0.0, min(1.0, 1.0 - (avg_risk - 0.7)))

                feedback_weight = min(1.0, feedback_count / max(float(min_feedback), 1.0))
                exploration_bonus = min(0.08, 0.01 * event_count)
                score = (
                    0.35 * confidence_score
                    + 0.20 * safety_score
                    + 0.25 * rating_score * feedback_weight
                    + 0.20 * success_score * feedback_weight
                    + exploration_bonus
                )
                strategy_scores[strategy] = round(float(score), 6)

            if total_feedback < min_feedback:
                return {
                    "strategy": None,
                    "reason": "insufficient-feedback",
                    "context": {
                        "task_type": task_type,
                        "adapter_type": adapter_type,
                        "model_size_bucket": model_size_bucket,
                    },
                    "feedback_samples": total_feedback,
                    "strategy_scores": strategy_scores,
                }

            best_strategy = max(strategy_scores.items(), key=lambda item: item[1])[0]
            return {
                "strategy": best_strategy,
                "reason": "feedback-policy",
                "context": {
                    "task_type": task_type,
                    "adapter_type": adapter_type,
                    "model_size_bucket": model_size_bucket,
                },
                "feedback_samples": total_feedback,
                "strategy_scores": strategy_scores,
            }

    def list_configs(
        self,
        task_type: str | None = None,
        model_size_bucket: str | None = None,
        adapter_type: str | None = None,
        limit: int = 200,
    ) -> list[dict]:
        with self.session_scope() as session:
            stmt = select(NormalizedConfigRow).order_by(NormalizedConfigRow.updated_at.desc())
            if task_type:
                stmt = stmt.where(NormalizedConfigRow.task_type == task_type)
            if model_size_bucket:
                stmt = stmt.where(NormalizedConfigRow.model_size_bucket == model_size_bucket)
            if adapter_type:
                stmt = stmt.where(NormalizedConfigRow.adapter_type == adapter_type)
            rows = session.execute(stmt.limit(max(1, min(limit, 1000)))).scalars().all()
            return [row.payload for row in rows]


def default_database_url() -> str:
    sqlite_path = Path("artifacts") / "ft_config_engine.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path.resolve()}"


def resolve_database_url() -> str:
    return normalize_database_url(os.environ.get("DATABASE_URL", default_database_url()))


def normalize_database_url(database_url: str) -> str:
    url = database_url.strip()
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url
