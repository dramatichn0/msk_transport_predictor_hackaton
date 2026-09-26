"""Validated, causal contracts shared by backend and ML services."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_BATCH = 256


def utc_time(value: str | datetime) -> datetime:
    """Convert a timestamp to timezone-aware UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return (parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None
            else parsed.astimezone(timezone.utc))


def estimate_dwell_seconds(history: list[dict]) -> int:
    """Estimate bounded continuous low-speed duration from timestamped samples."""
    stopped_reversed = []
    for point in reversed(history):
        if float(point.get("speed") or 0) >= 1:
            break
        stopped_reversed.append(point)
    stopped = list(reversed(stopped_reversed))
    if len(stopped) < 2:
        return 0
    try:
        start = utc_time(stopped[0]["event_time"])
        end = utc_time(stopped[-1]["event_time"])
    except (KeyError, TypeError, ValueError):
        return 0
    return min(300, max(0, int((end - start).total_seconds())))


class FiniteModel(BaseModel):
    """Reject NaN and infinity at the API boundary."""

    model_config = ConfigDict(allow_inf_nan=False)


class HistoryPoint(FiniteModel):
    """One historical observation available at prediction time."""

    event_time: datetime
    speed: float | None = Field(default=None, ge=0, le=180)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    heading: float | None = Field(default=None, ge=0, lt=360)
    door_open: bool | None = None
    door_passenger_count: int = Field(default=0, ge=0)

    @field_validator("event_time")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        """Normalize historical timestamps to UTC."""
        return utc_time(value)


class PredictionInput(FiniteModel):
    """Point-in-time prediction input for an official scheduled stop."""

    tr_id: str = Field(min_length=1, max_length=128)
    event_time: datetime
    target_time_begin: datetime
    target_stop_id: str = Field(min_length=1, max_length=128)
    cur_dev_s: float | None = None
    speed: float = Field(default=0, ge=0, le=180)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    heading: float | None = Field(default=None, ge=0, lt=360)
    stop_distance_m: float = Field(default=0, ge=0)
    nearby_context: int = Field(default=0, ge=0)
    target_stop_lat: float | None = Field(default=None, ge=-90, le=90)
    target_stop_lon: float | None = Field(default=None, ge=-180, le=180)
    target_stop_index: int = Field(default=0, ge=0)
    route_stop_count: int = Field(default=1, ge=1)
    target_progress: float = Field(default=0, ge=0, le=1)
    segment_index: int = Field(default=0, ge=0)
    segment_progress: float = Field(default=0, ge=0, le=1)
    segment_distance_m: float = Field(default=0, ge=0)
    heading_error_deg: float = Field(default=0, ge=0, le=180)
    door_open_events: int = Field(default=0, ge=0)
    stopped_seconds: int = Field(default=0, ge=0)
    current_progress: float = Field(default=0, ge=0, le=1)
    segment_avg_speed_kmh: float = Field(default=0, ge=0, le=180)
    speed_drop_kmh: float = Field(default=0, ge=0, le=180)
    door_passenger_count: int = Field(default=0, ge=0)
    history: list[HistoryPoint] = Field(default_factory=list, max_length=30)

    @field_validator("event_time", "target_time_begin")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        """Normalize all inference timestamps to UTC."""
        return utc_time(value)

    @model_validator(mode="after")
    def validate_causal_window(self) -> PredictionInput:
        """Enforce the strict 10–15 minute target horizon and causal history."""
        seconds = (self.target_time_begin - self.event_time).total_seconds()
        if not 600 < seconds <= 900:
            raise ValueError("target_time_begin must be strictly in (T+10 min, T+15 min]")
        if any(point.event_time > self.event_time for point in self.history):
            raise ValueError("history must contain only event_time <= T")
        self.history.sort(key=lambda point: point.event_time)
        return self


class BatchPredictionInput(BaseModel):
    """Bounded batch prediction request."""

    predictions: list[PredictionInput] = Field(min_length=1, max_length=MAX_BATCH)


class PredictionOutput(FiniteModel):
    """Prediction response with explicit degraded-mode semantics."""

    delay_seconds: float | None
    delay_probability: float | None = Field(default=None, ge=0, le=1)
    probability_source: Literal["classifier", "unavailable"] = "unavailable"
    model: str
    horizon_minutes: float = Field(gt=10, le=15)
    target_stop_id: str
    degraded: bool = False
    reason: str | None = None
    absolute_error_seconds: float | None = None


class BatchPredictionOutput(BaseModel):
    """Batch response validated by the backend before state mutation."""

    predictions: list[PredictionOutput]
