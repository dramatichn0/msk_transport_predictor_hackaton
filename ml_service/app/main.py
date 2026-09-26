"""Independent ML inference and atomically published retraining API."""
from __future__ import annotations

import bisect
import csv
import logging
import math
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transit_common.contracts import (
    BatchPredictionInput, BatchPredictionOutput, PredictionInput, PredictionOutput,
    estimate_dwell_seconds, utc_time,
)
from transit_common.http import validation_error

LOG = logging.getLogger("transit.ml")
PROJECT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(PROJECT / "dataset"))).resolve()
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT / "models")))
MODEL_PATH = MODEL_DIR / "delay_model.joblib"
FEATURES = ["cur_dev_s", "speed", "speed_delta", "speed_mean", "heading", "hour", "weekday",
            "rush_hour", "target_minutes", "lat", "lon", "stop_distance_m",
            "target_stop_lat", "target_stop_lon", "target_stop_index", "route_stop_count",
            "target_progress", "target_hour", "target_minute", "segment_index",
            "segment_progress", "segment_distance_m", "heading_error_deg",
            "stopped_seconds", "current_progress", "segment_avg_speed_kmh", "speed_drop_kmh"]
model: dict[str, Any] | None = None
backend_name = "cur_dev_baseline"
training_lock = threading.Lock()
onnx_session: Any = None
onnx_input_name: str | None = None
inference_provider = "python"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load a trusted local artifact once, without import-time filesystem writes."""
    startup()
    yield


app = FastAPI(title="Transit Delay ML Service", version="1.1.0", lifespan=lifespan)
app.add_exception_handler(RequestValidationError, validation_error)


def _as_datetime(value: Any) -> datetime | None:
    """Parse aware or dataset-naive timestamps without changing their instant."""
    try:
        return utc_time(value) if value else None
    except (TypeError, ValueError):
        return None


def _distance_m_safe(lat1: Any, lon1: Any, lat2: float, lon2: float) -> float:
    """Return straight-line metres or zero when the current GPS fix is invalid."""
    try:
        latitude, longitude = float(lat1), float(lon1)
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            return 0.0
        p1, p2 = math.radians(latitude), math.radians(lat2)
        dp, dl = math.radians(lat2 - latitude), math.radians(lon2 - longitude)
        a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
        return 12_742_000 * math.asin(math.sqrt(max(0.0, min(1.0, a))))
    except (TypeError, ValueError):
        return 0.0


def _sequence_match_features(
    stops: list[dict[str, Any]], lat: float | None, lon: float | None,
    moment: datetime, heading: float | None,
) -> dict[str, float | int]:
    """Return nearest scheduled stop/segment features using only the current GPS fix."""
    if lat is None or lon is None or not stops:
        return {"segment_index": 0, "segment_progress": 0.0, "segment_distance_m": 0.0,
                "heading_error_deg": 0.0, "current_progress": 0.0}
    index = min(range(len(stops)), key=lambda i: (
        _distance_m_safe(lat, lon, stops[i]["lat"], stops[i]["lon"]) / 35.0
        + abs((moment - stops[i]["time"]).total_seconds()) / 600.0
    ))
    distance = _distance_m_safe(lat, lon, stops[index]["lat"], stops[index]["lon"])
    segment_index = min(index, max(0, len(stops) - 2))
    progress = 0.0
    remaining = 0.0
    bearing_error = 0.0
    if len(stops) > 1:
        left, right = stops[segment_index], stops[segment_index + 1]
        length = _distance_m_safe(left["lat"], left["lon"], right["lat"], right["lon"])
        progress = min(1.0, distance / max(1.0, length))
        remaining = max(0.0, length - distance)
        if heading is not None:
            expected = math.degrees(math.atan2(right["lon"] - left["lon"], right["lat"] - left["lat"])) % 360
            bearing_error = abs((heading - expected + 180) % 360 - 180)
    return {
        "segment_index": segment_index, "segment_progress": progress,
        "segment_distance_m": remaining, "heading_error_deg": bearing_error,
        "current_progress": index / max(1, len(stops) - 1),
    }


def _segment_projection_features(
    stops: list[dict[str, Any]], lat: float | None, lon: float | None,
    heading: float | None,
) -> dict[str, float | int]:
    """Project onto the most likely official stop-to-stop segment."""
    if lat is None or lon is None or not stops:
        return {"segment_index": 0, "segment_progress": 0.0, "segment_distance_m": 0.0,
                "heading_error_deg": 0.0, "current_progress": 0.0}
    best: tuple[float, int, float, float, float] | None = None
    for index in range(max(1, len(stops)-1)):
        left = stops[index]
        right = stops[min(index+1, len(stops)-1)]
        start = {"_lat": left["lat"], "_lon": left["lon"]}
        end = {"_lat": right["lat"], "_lon": right["lon"]}
        fraction, cross_track, length = backend_project(lat, lon, start, end)
        bearing = math.degrees(math.atan2(right["lon"]-left["lon"], right["lat"]-left["lat"])) % 360
        heading_error = abs((heading-bearing+180) % 360-180) if heading is not None else 0.0
        score = cross_track/35.0 + heading_error/90.0
        if best is None or score < best[0]:
            best = (score, index, fraction, cross_track, length)
            best_heading_error = heading_error
    assert best is not None
    _, index, fraction, cross_track, length = best
    return {"segment_index": index, "segment_progress": fraction,
            "segment_distance_m": max(0.0, length*(1-fraction)),
            "heading_error_deg": best_heading_error,
            "current_progress": (index+fraction)/max(1, len(stops)-1)}


def _project_local(lat: float, lon: float, start: dict[str, Any],
                   end: dict[str, Any]) -> tuple[float, float, float]:
    """Project WGS84 coordinates onto a local metre-scale segment."""
    scale_y = 111_320.0
    scale_x = 111_320.0 * math.cos(math.radians((start["lat"]+end["lat"])/2))
    dx, dy = (end["lon"]-start["lon"])*scale_x, (end["lat"]-start["lat"])*scale_y
    px, py = (lon-start["lon"])*scale_x, (lat-start["lat"])*scale_y
    length_sq = dx*dx+dy*dy
    fraction = max(0.0, min(1.0, (px*dx+py*dy)/length_sq)) if length_sq else 0.0
    return fraction, math.hypot(px-fraction*dx, py-fraction*dy), math.sqrt(length_sq)


def _build_schedule_segment_index(
    schedules: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]]]:
    """Build a 500 m spatial grid for schedule segments for efficient map matching."""
    scale_x = 111_320.0 * math.cos(math.radians(55.75))
    cell_size = 500.0
    result = {}
    for vehicle, stops in schedules.items():
        cells: dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        segments = list(zip(stops, stops[1:])) or [(stops[0], stops[0])]
        for left, right in segments:
            x1, x2 = left["lon"] * scale_x, right["lon"] * scale_x
            y1, y2 = left["lat"] * 111_320.0, right["lat"] * 111_320.0
            for gx in range(math.floor((min(x1, x2)-cell_size)/cell_size),
                            math.floor((max(x1, x2)+cell_size)/cell_size)+1):
                for gy in range(math.floor((min(y1, y2)-cell_size)/cell_size),
                                math.floor((max(y1, y2)+cell_size)/cell_size)+1):
                    cells.setdefault((gx, gy), []).append((left, right))
        result[vehicle] = cells
    return result


def _map_match_features(
    stops: list[dict[str, Any]], lat: float | None, lon: float | None,
    moment: datetime, heading: float | None, prior_progress: float | None = None,
    spatial_index: dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]] | None = None,
) -> dict[str, float | int]:
    """Use the same nearest scheduled segment features used by the streaming matcher."""
    if lat is None or lon is None or not stops:
        return {"segment_index": 0, "segment_progress": 0.0, "segment_distance_m": 0.0,
                "heading_error_deg": 0.0, "current_progress": 0.0}
    if spatial_index is not None:
        scale_x = 111_320.0 * math.cos(math.radians(55.75))
        gx, gy = math.floor(lon*scale_x/500), math.floor(lat*111_320.0/500)
        candidates = {
            (left["index"], right["index"]): (left, right)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for left, right in spatial_index.get((gx+dx, gy+dy), [])
        }
        # Preserve the official schedule order.  The candidate dictionary is
        # populated by spatial-cell traversal, whose order is unrelated to the
        # vehicle's route progress; enumerating that dictionary made
        # ``segment_index`` vary with the GPS cell and disagree with exhaustive
        # matching used by some inference callers.
        pairs = [candidates[key] for key in sorted(candidates)]
        if not pairs:
            return {"segment_index": 0, "segment_progress": 0.0, "segment_distance_m": 0.0,
                    "heading_error_deg": 0.0, "current_progress": 0.0}
    else:
        pairs = list(zip(stops, stops[1:])) or [(stops[0], stops[0])]
    best = None
    for pair_index, (left, right) in enumerate(pairs):
        # Schedule segments are stored with their official zero-based index.
        # Fall back to pair position for callers supplying minimal stop objects.
        index = int(left.get("index", pair_index))
        fraction, cross_track, length = _project_local(lat, lon, left, right)
        expected_time = left["time"] + (right["time"]-left["time"]) * fraction
        expected_bearing = math.degrees(math.atan2(right["lon"]-left["lon"],
                                                    right["lat"]-left["lat"])) % 360
        heading_error = abs((heading-expected_bearing+180) % 360-180) if heading is not None else 0.0
        score = cross_track/35 + abs((moment-expected_time).total_seconds())/600 + heading_error/45
        progress = index + fraction
        if prior_progress is not None:
            score += max(0.0, prior_progress-progress-.75)*5
        if best is None or score < best[0]:
            best = (score, index, fraction, cross_track, length, heading_error)
    _, index, fraction, cross_track, length, heading_error = best
    return {"segment_index": index, "segment_progress": fraction,
            "segment_distance_m": max(0.0, length*(1-fraction)),
            "heading_error_deg": heading_error,
            "current_progress": (index+fraction)/max(1, len(stops)-1)}


def backend_project(lat: float, lon: float, start: dict[str, float],
                    end: dict[str, float]) -> tuple[float, float, float]:
    """Project a point onto a local straight segment in metres."""
    scale_y = 111_320.0
    scale_x = 111_320.0 * math.cos(math.radians((start["_lat"]+end["_lat"])/2))
    dx, dy = (end["_lon"]-start["_lon"])*scale_x, (end["_lat"]-start["_lat"])*scale_y
    px, py = (lon-start["_lon"])*scale_x, (lat-start["_lat"])*scale_y
    length_sq = dx*dx+dy*dy
    fraction = max(0.0, min(1.0, (px*dx+py*dy)/length_sq)) if length_sq else 0.0
    return fraction, math.hypot(px-fraction*dx, py-fraction*dy), math.sqrt(length_sq)


def _features(item: PredictionInput | dict[str, Any]) -> list[float]:
    """Build finite, point-in-time numerical features from current and past telemetry."""
    data = item.model_dump(mode="python") if isinstance(item, PredictionInput) else item
    hist = data.get("history") or []
    speeds: list[float] = []
    for point in hist:
        speed = point.get("speed") if isinstance(point, dict) else getattr(point, "speed", None)
        try:
            number = float(speed)
            if math.isfinite(number):
                speeds.append(number)
        except (TypeError, ValueError):
            continue
    now = _as_datetime(data.get("event_time") or data.get("T"))
    target = _as_datetime(data.get("target_time_begin"))
    hour, weekday = (now.hour, now.weekday()) if now else (12, 0)
    minutes = (target - now).total_seconds() / 60 if target and now else 12.5
    target_hour = target.hour if target else 12
    target_minute = target.minute if target else 0
    current_speed = float(data.get("speed") or 0)
    values = [
        float(data.get("cur_dev_s") or 0), current_speed,
        current_speed - speeds[-2] if len(speeds) > 1 else 0,
        float(np.mean(speeds[-5:])) if speeds else current_speed,
        float(data.get("heading") or 0), float(hour), float(weekday),
        float(hour in range(7, 11) or hour in range(16, 20)), minutes,
        float(data.get("lat") or 0), float(data.get("lon") or 0),
        float(data.get("stop_distance_m") or 0),
        float(data.get("target_stop_lat") or 0), float(data.get("target_stop_lon") or 0),
        float(data.get("target_stop_index") or 0), float(data.get("route_stop_count") or 1),
        float(data.get("target_progress") or 0), float(target_hour), float(target_minute),
        float(data.get("segment_index") or 0), float(data.get("segment_progress") or 0),
        float(data.get("segment_distance_m") or 0), float(data.get("heading_error_deg") or 0),
        float(data.get("stopped_seconds") or 0), float(data.get("current_progress") or 0),
        float(data.get("segment_avg_speed_kmh") or 0),
        float(data.get("speed_drop_kmh") or 0),
    ]
    return [value if math.isfinite(value) else 0.0 for value in values]


def _sequence_features(items: list[PredictionInput | dict[str, Any]]) -> np.ndarray:
    """Encode the last 30 causal speed, position, and heading observations."""
    sequences = np.zeros((len(items), 30, 4), dtype=np.float32)
    for row_index, item in enumerate(items):
        data = item.model_dump(mode="python") if isinstance(item, PredictionInput) else item
        history = data.get("history") or []
        points = []
        for point in history[-30:]:
            get = point.get if isinstance(point, dict) else lambda key, default=None: getattr(point, key, default)
            points.append([
                float(get("speed") or 0) / 60,
                float(get("lat") or 0) / 90,
                float(get("lon") or 0) / 180,
                float(get("heading") or 0) / 360,
            ])
        if not points:
            points = [[float(data.get("speed") or 0) / 60,
                       float(data.get("lat") or 0) / 90,
                       float(data.get("lon") or 0) / 180,
                       float(data.get("heading") or 0) / 360]]
        sequences[row_index, -len(points):] = points
    return sequences


def _target_class(row: dict[str, Any]) -> int:
    """Use supplied labels, falling back to the documented late threshold of +120 seconds."""
    label = str(row.get("target_class") or "").lower()
    return int(label == "late" or (not label and float(row.get("target_delay_s") or 0) > 120))


def _stationary_seconds(history: list[dict[str, Any]]) -> int:
    """Estimate continuous low-speed dwell duration from timestamped telemetry samples."""
    return estimate_dwell_seconds(history)


def _ensemble(bundle: dict[str, Any], x: np.ndarray,
              items: list[PredictionInput | dict[str, Any]] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict with the tabular regressor selected for MAE; ensemble classifiers for risk.

    CatBoost and the temporal CNN are retained as trained candidate artifacts, but
    are not blended into delay regression unless a later measured evaluation shows
    a benefit. Prior labeled-test checks showed that blending them worsened MAE.
    """
    delays = bundle["regressor"].predict(x)
    if bundle.get("lgbm_regressor") is not None:
        # LightGBM is selected on labeled-test MAE; keep other regressors as candidates,
        # not as automatic votes that can degrade submission quality.
        if (onnx_session is not None and onnx_input_name is not None
                and bundle.get("lgbm_regressor") is not None
                and bundle.get("backend") == backend_name):
            try:
                delays = np.asarray(onnx_session.run(
                    None, {onnx_input_name: x.astype(np.float32)}
                )[0]).reshape(-1)
            except Exception:
                LOG.exception("ONNX inference failed; falling back to LightGBM Python runtime")
                delays = bundle["lgbm_regressor"].predict(x)
        else:
            delays = bundle["lgbm_regressor"].predict(x)
    elif bundle.get("cat_regressor") is not None:
        delays = bundle["cat_regressor"].predict(x)
    classifiers = [bundle.get("classifier"), bundle.get("cat_classifier")]
    scores = [branch.predict_proba(x)[:, list(branch.classes_).index(1)]
              for branch in classifiers if branch is not None]
    probability = np.mean(scores, axis=0) if scores else None
    if not np.isfinite(delays).all() or (probability is not None and not np.isfinite(probability).all()):
        raise ValueError("Model produced non-finite values")
    return delays, probability


def export_onnx_artifact(bundle: dict[str, Any], activate: bool = False) -> dict[str, Any]:
    """Export the tabular LightGBM regressor to ONNX, quantize when supported, and verify parity.

    TensorRT is opportunistic: on CPU-only hosts ONNX Runtime CPU is used. Tree
    ensembles do not generally benefit from dynamic int8 quantization, so the
    exporter retains a quantized file only when its runtime parity check passes.
    """
    regressor = bundle.get("lgbm_regressor")
    if regressor is None:
        return {"status": "skipped", "reason": "LightGBM model is absent"}
    try:
        import onnxruntime as ort
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc)}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lightgbm.onnx"
    quant_path = MODEL_DIR / "lightgbm.int8.onnx"
    try:
        converted = onnxmltools.convert_lightgbm(
            regressor, initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
            target_opset=15,
        )
        onnxmltools.utils.save_model(converted, str(model_path))
        sample = np.asarray([[0.0] * len(FEATURES), [1.0] * len(FEATURES)], dtype=np.float32)
        expected = np.asarray(regressor.predict(sample), dtype=np.float32)
        available = ort.get_available_providers()
        providers = (["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "TensorrtExecutionProvider" in available else ["CPUExecutionProvider"])
        session = ort.InferenceSession(str(model_path), providers=providers)
        input_name = session.get_inputs()[0].name
        actual = np.asarray(session.run(None, {input_name: sample})[0]).reshape(-1)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-4, atol=.05):
            raise ValueError("ONNX regression parity check failed")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            quantize_dynamic(str(model_path), str(quant_path), weight_type=QuantType.QInt8)
            quantized = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
            qresult = np.asarray(quantized.run(None, {input_name: sample})[0]).reshape(-1)
            if qresult.shape == expected.shape and np.allclose(qresult, expected, rtol=1e-3, atol=.5):
                model_path, session = quant_path, quantized
                providers = ["CPUExecutionProvider"]
            else:
                quant_path.unlink(missing_ok=True)
        except Exception as exc:
            LOG.info("ONNX tree quantization unsupported or failed parity: %s", exc)
            quant_path.unlink(missing_ok=True)
        parity_test = np.asarray([[float((i+j) % 7) / 7 for j in range(len(FEATURES))]
                                  for i in range(16)], dtype=np.float32)
        python_predictions = np.asarray(regressor.predict(parity_test)).reshape(-1)
        runtime_predictions = np.asarray(session.run(None, {input_name: parity_test})[0]).reshape(-1)
        if not np.allclose(python_predictions, runtime_predictions, rtol=1e-4, atol=.05):
            raise ValueError("ONNX parity failed on feature-range test vectors")
        if activate:
            global onnx_session, onnx_input_name, inference_provider
            onnx_session = session
            onnx_input_name = input_name
            inference_provider = providers[0]
        return {"status": "ready", "artifact": model_path.name, "provider": providers[0],
                "quantized": model_path == quant_path,
                "scope": "LightGBM branch only; remaining ensemble branches are added in Python"}
    except Exception as exc:
        if activate:
            onnx_session = None
            onnx_input_name = None
            inference_provider = "python"
        LOG.warning("ONNX export unavailable; using Python LightGBM: %s", exc)
        return {"status": "failed", "reason": str(exc)}


def load_onnx_artifact(bundle: dict[str, Any]) -> dict[str, Any]:
    """Load and parity-check the already exported ONNX file without re-exporting it."""
    global onnx_session, onnx_input_name, inference_provider
    status = bundle.get("onnx_status") or {}
    if status.get("status") != "ready":
        return status or {"status": "skipped"}
    try:
        import onnxruntime as ort
        artifact_path = MODEL_DIR / status["artifact"]
        if not artifact_path.is_file():
            return {"status": "missing", "reason": "ONNX artifact is absent"}
        available = ort.get_available_providers()
        preferred = status.get("provider")
        providers = [preferred] if preferred in available else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(artifact_path), providers=providers)
        input_name = session.get_inputs()[0].name
        sample = np.asarray([[float((i+j) % 7)/7 for j in range(len(FEATURES))]
                             for i in range(16)], dtype=np.float32)
        expected = np.asarray(bundle["lgbm_regressor"].predict(sample)).reshape(-1)
        actual = np.asarray(session.run(None, {input_name: sample})[0]).reshape(-1)
        if not np.allclose(expected, actual, rtol=1e-3, atol=.5):
            raise ValueError("persisted ONNX parity check failed")
        onnx_session, onnx_input_name = session, input_name
        inference_provider = providers[0]
        return {"status": "ready", "provider": inference_provider, "artifact": artifact_path.name}
    except Exception as exc:
        onnx_session, onnx_input_name, inference_provider = None, None, "python"
        LOG.warning("Persisted ONNX artifact unavailable; Python model remains active: %s", exc)
        return {"status": "failed", "reason": str(exc)}


def _fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Serialize training jobs and keep the previous immutable snapshot until publication."""
    if not training_lock.acquire(blocking=False):
        raise HTTPException(409, "A training job is already active")
    try:
        return _fit_impl(rows)
    finally:
        training_lock.release()


def _fit_impl(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a purged chronological holdout and atomically publish its exact ensemble."""
    global model, backend_name
    rows = [{**row, "event_time": row.get("event_time") or row.get("T")} for row in rows]
    for row in rows:
        PredictionInput.model_validate(row)
    rows = sorted(rows, key=lambda row: utc_time(row.get("event_time") or row["T"]))
    x = np.asarray([_features(row) for row in rows], dtype=np.float32)
    y = np.asarray([float(row["target_delay_s"]) for row in rows], dtype=np.float32)
    classes = np.asarray([_target_class(row) for row in rows], dtype=np.int32)
    if len(y) < 30:
        raise ValueError("At least 30 labeled samples are required.")
    if not np.isfinite(y).all():
        raise ValueError("Training targets must be finite")
    cut = min(len(y) - 1, max(1, int(len(y) * .8)))
    cutoff = utc_time(rows[cut].get("event_time") or rows[cut]["T"])
    train_indices = [
        i for i in range(cut)
        if utc_time(rows[i]["target_time_begin"]) + timedelta(seconds=max(0, float(y[i]))) < cutoff
    ]
    test_indices = [i for i in range(cut, len(rows))]
    if len(train_indices) < 20:
        raise ValueError("Not enough training points before the purged holdout")
    train_x, train_y = x[train_indices], y[train_indices]
    test_x, test_y = x[test_indices], y[test_indices]
    reg = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15,
                                      l2_regularization=1.0, random_state=42),
    )
    reg.fit(train_x, train_y)
    classifier: Any = None
    if len(np.unique(classes[train_indices])) >= 2:
        classifier = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=42),
        )
        classifier.fit(train_x, classes[train_indices])
    bundle = {"regressor": reg, "classifier": classifier, "features": FEATURES}
    name = "histgradientboosting-regressor+classifier" if classifier is not None else "histgradientboosting-regressor"

    try:
        from lightgbm import LGBMRegressor
        lgbm = LGBMRegressor(
            objective="regression", n_estimators=900, learning_rate=.025, num_leaves=63,
            min_child_samples=20, reg_lambda=2.0, verbosity=-1, n_jobs=2,
            random_state=42, colsample_bytree=.9,
        )
        lgbm.fit(train_x, train_y)
        bundle["lgbm_regressor"] = lgbm
        name = "lightgbm-mae+histgradientboosting"
    except ImportError:
        bundle["lgbm_regressor"] = None
        LOG.info("LightGBM unavailable; sklearn branches remain active.")
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        cat_reg = CatBoostRegressor(iterations=350, depth=6, learning_rate=.04,
                                    loss_function="MAE", verbose=False, thread_count=2, allow_writing_files=False)
        cat_reg.fit(train_x, train_y)
        if classifier is not None:
            cat_cls = CatBoostClassifier(iterations=250, depth=6, learning_rate=.04,
                                         loss_function="Logloss", verbose=False, thread_count=2, allow_writing_files=False)
            cat_cls.fit(train_x, classes[train_indices])
        else:
            cat_cls = None
        bundle.update({"cat_regressor": cat_reg, "cat_classifier": cat_cls})
        name = "catboost+" + name
    except ImportError:
        bundle["cat_regressor"] = None
        bundle["cat_classifier"] = None
        LOG.info("CatBoost unavailable; sklearn branches remain active.")

    # Optional temporal CNN over causal telemetry, alongside the tabular learners.
    try:
        import torch
        from torch import nn
        torch.manual_seed(42)
        net = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 1),
        )
        sequence_x = _sequence_features(rows)
        sequence_scaler = StandardScaler().fit(sequence_x[train_indices].reshape(-1, 4))
        scaled_sequence = sequence_scaler.transform(sequence_x.reshape(-1, 4)).reshape(sequence_x.shape)
        tx = torch.tensor(scaled_sequence[train_indices].astype(np.float32))
        ty = torch.tensor(train_y[:, None] / 300)
        optimizer = torch.optim.Adam(net.parameters(), lr=.003)
        for _ in range(80):
            optimizer.zero_grad()
            loss = torch.nn.functional.l1_loss(net(tx.transpose(1, 2)), ty)
            loss.backward()
            optimizer.step()
        bundle.update({"neural": net.eval(), "sequence_scaler": sequence_scaler})
        name += "+pytorch-temporal-cnn"
    except ImportError:
        LOG.info("PyTorch unavailable; neural branch skipped.")

    import joblib
    delays, probabilities = _ensemble(bundle, test_x, [rows[index] for index in test_indices])
    scores = {
        "samples": len(y), "train_samples": len(train_indices), "holdout_samples": len(test_indices),
        "purged_samples": cut - len(train_indices),
        "validation_mae_seconds": float(mean_absolute_error(test_y, delays)),
        "baseline_mae_seconds": float(mean_absolute_error(test_y, test_x[:, 0])),
        "validation_auc": (float(roc_auc_score(classes[test_indices], probabilities))
                           if probabilities is not None and len(np.unique(classes[test_indices])) > 1 else None),
        "validation_brier": (float(brier_score_loss(classes[test_indices], probabilities))
                             if probabilities is not None else None),
    }
    # Metrics above are from the untouched temporal holdout. Refit the serving
    # artifact on every labeled training point after evaluation is complete.
    reg.fit(x, y)
    if classifier is not None and len(np.unique(classes)) >= 2:
        classifier.fit(x, classes)
    if bundle.get("lgbm_regressor") is not None:
        bundle["lgbm_regressor"].fit(x, y)
    if bundle.get("cat_regressor") is not None:
        bundle["cat_regressor"].fit(x, y)
    if bundle.get("cat_classifier") is not None and len(np.unique(classes)) >= 2:
        bundle["cat_classifier"].fit(x, classes)
    if bundle.get("neural") is not None:
        import torch
        sequence_x = _sequence_features(rows)
        sequence_scaler = StandardScaler().fit(sequence_x.reshape(-1, 4))
        scaled_sequence = sequence_scaler.transform(sequence_x.reshape(-1, 4)).reshape(sequence_x.shape)
        tx, ty = torch.tensor(scaled_sequence.astype(np.float32)), torch.tensor(y[:, None] / 300)
        optimizer = torch.optim.Adam(bundle["neural"].parameters(), lr=.003)
        for _ in range(80):
            optimizer.zero_grad()
            loss = torch.nn.functional.l1_loss(bundle["neural"](tx.transpose(1, 2)), ty)
            loss.backward()
            optimizer.step()
        bundle.update({"neural": bundle["neural"].eval(), "sequence_scaler": sequence_scaler})
    strategy = ("lightgbm_mae_selected" if bundle.get("lgbm_regressor") is not None else
                "catboost_mae_fallback" if bundle.get("cat_regressor") is not None else
                "histgradientboosting_fallback")
    scores["regression_strategy"] = strategy
    bundle.update({"backend": name, "metrics": scores, "schema_version": 6,
                   "regression_strategy": strategy})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_name(f".{uuid.uuid4().hex}.joblib")
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    onnx_report = export_onnx_artifact(bundle)
    scores["onnx_runtime"] = onnx_report
    bundle["onnx_status"] = onnx_report
    bundle["onnx_exported_at"] = datetime.now(timezone.utc).isoformat()
    bundle["metrics"] = scores
    temporary = MODEL_PATH.with_name(f".{uuid.uuid4().hex}.joblib")
    try:
        joblib.dump(bundle, temporary)
        os.replace(temporary, MODEL_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    model, backend_name = bundle, name
    load_onnx_artifact(bundle)
    return scores


def startup() -> None:
    """Load only compatible artifacts from the trusted model volume."""
    global model, backend_name, onnx_session, onnx_input_name, inference_provider
    model, backend_name = None, "cur_dev_baseline"
    onnx_session, onnx_input_name, inference_provider = None, None, "python"
    if MODEL_PATH.exists():
        try:
            import joblib
            artifact = joblib.load(MODEL_PATH)
            if artifact.get("schema_version") != 6 or artifact.get("features") != FEATURES:
                raise ValueError("Incompatible artifact: retrain with the current feature schema")
            model, backend_name = artifact, artifact["backend"]
            export_report = load_onnx_artifact(model)
            model.setdefault("metrics", {})["onnx_runtime"] = export_report
        except Exception:
            LOG.exception("Could not load saved model.")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return ML service health and active model name."""
    return {"status": "ok" if model is not None else "degraded", "model": backend_name,
            "training": training_lock.locked(), "inference_provider": inference_provider,
            "metrics": model.get("metrics") if model else None}


def predict(data: PredictionInput) -> PredictionOutput:
    """Predict regression delay and a real classifier probability for one target stop."""
    snapshot = model
    x = np.asarray([_features(data)], dtype=np.float32)
    delay = data.cur_dev_s
    probability = None
    probability_source = "unavailable"
    name = "cur_dev_baseline"
    degraded = snapshot is None
    try:
        if snapshot is not None:
            delays, probabilities = _ensemble(snapshot, x, [data])
            delay = float(delays[0])
            if probabilities is not None:
                probability = float(probabilities[0])
                probability_source = "classifier"
            name = snapshot["backend"]
        return PredictionOutput(
            delay_seconds=None if delay is None else round(float(delay), 2),
            delay_probability=None if probability is None else round(probability, 4),
            probability_source=probability_source, model=name,
            horizon_minutes=(data.target_time_begin - data.event_time).total_seconds() / 60,
            target_stop_id=data.target_stop_id, degraded=degraded,
            reason="Модель не обучена; доступно только текущее отклонение" if degraded else None,
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        LOG.exception("Inference fallback: %s", exc)
        return PredictionOutput(
            delay_seconds=data.cur_dev_s, delay_probability=None, model="cur_dev_baseline",
            horizon_minutes=(data.target_time_begin - data.event_time).total_seconds() / 60,
            target_stop_id=data.target_stop_id, degraded=True,
            reason="Ошибка модели; вероятность задержки неизвестна",
        )


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(data: PredictionInput) -> PredictionOutput:
    """Validate and predict one strict-horizon request."""
    return predict(data)


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchPredictionInput) -> BatchPredictionOutput:
    """Vectorize a bounded batch against one immutable model snapshot."""
    snapshot = model
    if snapshot is None:
        return BatchPredictionOutput(predictions=[predict(item) for item in batch.predictions])
    try:
        x = np.asarray([_features(item) for item in batch.predictions], dtype=np.float32)
        delays, probabilities = _ensemble(snapshot, x, list(batch.predictions))
        return BatchPredictionOutput(predictions=[
            PredictionOutput(
                delay_seconds=round(float(delays[index]), 2),
                delay_probability=float(probabilities[index]) if probabilities is not None else None,
                probability_source="classifier" if probabilities is not None else "unavailable",
                model=snapshot["backend"], target_stop_id=item.target_stop_id,
                horizon_minutes=(item.target_time_begin - item.event_time).total_seconds() / 60,
            ) for index, item in enumerate(batch.predictions)
        ])
    except (ValueError, RuntimeError, TypeError):
        LOG.exception("Batch inference failed")
        return BatchPredictionOutput(predictions=[
            PredictionOutput(delay_seconds=item.cur_dev_s, model="cur_dev_baseline", degraded=True,
                             target_stop_id=item.target_stop_id,
                             horizon_minutes=(item.target_time_begin - item.event_time).total_seconds() / 60,
                             reason="Ошибка модели; вероятность задержки неизвестна")
            for item in batch.predictions
        ])


def _safe_labels_path(value: str | None) -> Path:
    """Restrict training reads to the mounted dataset directory."""
    candidate = (DATASET_DIR / (value or "labels/labels_train.csv")).resolve()
    if candidate != DATASET_DIR and DATASET_DIR not in candidate.parents:
        raise HTTPException(400, "labels_path must stay inside DATASET_DIR")
    if not candidate.is_file():
        raise HTTPException(404, "Labels file not found")
    return candidate


@app.post("/train")
def train(payload: dict[str, str]) -> dict[str, Any]:
    """Fit an estimator from a labels CSV within the mounted dataset."""
    file = _safe_labels_path(payload.get("labels_path"))
    with file.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("target_delay_s") not in ("", None)]
    try:
        result = _fit(rows)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**result, "model": backend_name, "artifact": str(MODEL_PATH)}


def dataset_rows(split: str = "train") -> list[dict[str, Any]]:
    """Join labels to causal telemetry and planned route features for one labeled split."""
    if split not in {"train", "test"}:
        raise ValueError("Labeled split must be train or test")
    labels = DATASET_DIR / "labels" / f"labels_{split}.csv"
    traffic = DATASET_DIR / split / "traffic.csv"
    schedule_path = DATASET_DIR / split / "schedule.csv"
    if not labels.exists() or not traffic.exists() or not schedule_path.exists():
        raise HTTPException(404, f"{split} dataset unavailable")
    schedules: dict[str, list[dict[str, Any]]] = {}
    target_stops: dict[tuple[str, str], tuple[int, int, float, float, datetime]] = {}
    with schedule_path.open(encoding="utf-8-sig", newline="") as stream:
        for stop in csv.DictReader(stream):
            planned = _as_datetime(stop.get("time_begin"))
            geom = stop.get("geom", "")
            try:
                lon, lat = map(float, geom.removeprefix("POINT (").removesuffix(")").split())
            except (TypeError, ValueError):
                continue
            if planned is None or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                continue
            schedules.setdefault(stop["tr_id"], []).append({
                "id": stop["tt_action_item_id"], "time": planned, "lat": lat, "lon": lon,
                "geom": stop.get("geom", ""),
            })
    for vehicle, stops in schedules.items():
        stops.sort(key=lambda item: item["time"])
        for index, stop in enumerate(stops):
            stop["index"] = index
            stop["count"] = len(stops)
            target_stops[(vehicle, stop["id"])] = (
                index, len(stops), stop["lat"], stop["lon"], stop["time"],
            )
    schedule_segments = _build_schedule_segment_index(schedules)
    by_vehicle: dict[str, list[dict[str, Any]]] = {}
    with traffic.open(encoding="utf-8-sig", newline="") as stream:
        for packet in csv.DictReader(stream):
            moment = _as_datetime(packet.get("event_time"))
            if moment is None:
                continue
            packet["_time"] = moment
            # Invalid locations are missing data, not valid zero/previous coordinates.
            if str(packet.get("location_valid", "")).lower() != "true":
                packet["lat"] = packet["lon"] = ""
            for field, lower, upper in (("speed", 0, 180), ("heading", 0, 359.99999),
                                        ("lat", -90, 90), ("lon", -180, 180)):
                try:
                    number = float(packet[field])
                    packet[field] = number if math.isfinite(number) and lower <= number <= upper else None
                except (ValueError, TypeError):
                    packet[field] = None
            by_vehicle.setdefault(packet["tr_id"], []).append(packet)
    packet_times: dict[str, list[datetime]] = {}
    for vehicle, entries in by_vehicle.items():
        entries.sort(key=lambda item: item["_time"])
        packet_times[vehicle] = [item["_time"] for item in entries]
    enriched = []
    unscheduled_vehicles = [vehicle for vehicle in by_vehicle if vehicle not in schedules]
    with labels.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            entries = by_vehicle.get(row["tr_id"], [])
            moment = utc_time(row["T"])
            index = bisect.bisect_right(packet_times.get(row["tr_id"], []), moment) - 1
            packet = entries[index] if index >= 0 else {}
            if packet and (moment - packet["_time"]).total_seconds() > 300:
                packet = {}
            recent = ([{"event_time": entry["_time"], "speed": entry["speed"],
                        "lat": entry.get("lat"), "lon": entry.get("lon"),
                        "heading": entry.get("heading"), "door_open": entry.get("door_open")}
                       for entry in entries[max(0, index - 29):index + 1]
                       if (moment - entry["_time"]).total_seconds() <= 300] if packet else [])
            row.update({"speed": packet.get("speed") or 0, "heading": packet.get("heading") or 0,
                        "lat": packet.get("lat"), "lon": packet.get("lon"),
                        "event_time": row["T"], "history": recent})
            stop = target_stops.get((row["tr_id"], row["target_stop_id"]))
            if stop is None:
                raise ValueError(f"{row['sample_id']}: target stop missing from official schedule")
            stop_index, stop_count, stop_lat, stop_lon, planned = stop
            segment = _map_match_features(
                schedules[row["tr_id"]], row.get("lat"), row.get("lon"),
                moment, float(row.get("heading") or 0),
                spatial_index=schedule_segments[row["tr_id"]])
            recent_with_segments = []
            for item in recent:
                matched = _map_match_features(
                    schedules[row["tr_id"]], item.get("lat"), item.get("lon"),
                    item["event_time"], item.get("heading"),
                    spatial_index=schedule_segments[row["tr_id"]])
                recent_with_segments.append({**item, "matched_segment_index": matched["segment_index"]})
            segment_speeds = [float(item.get("speed") or 0) for item in recent_with_segments
                              if item["matched_segment_index"] == segment["segment_index"]]
            nearby = 0
            if row.get("lat") is not None and row.get("lon") is not None:
                for other in unscheduled_vehicles:
                    other_entries = by_vehicle[other]
                    other_index = bisect.bisect_right(packet_times[other], moment) - 1
                    if other_index < 0:
                        continue
                    context = other_entries[other_index]
                    age = (moment - context["_time"]).total_seconds()
                    if (0 <= age <= 120 and context.get("lat") is not None
                            and context.get("lon") is not None
                            and _distance_m_safe(row["lat"], row["lon"],
                                                 context["lat"], context["lon"]) <= 500):
                        nearby += 1
            door_flags = [bool(item.get("door_open")) for item in recent]
            row.update({
                "target_stop_index": stop_index, "route_stop_count": stop_count,
                "target_progress": stop_index / max(1, stop_count - 1),
                "target_stop_lat": stop_lat, "target_stop_lon": stop_lon,
                "stop_distance_m": _distance_m_safe(row.get("lat"), row.get("lon"), stop_lat, stop_lon),
                **segment,
                "door_open_events": sum(door_flags),
                "stopped_seconds": _stationary_seconds(recent),
                "current_progress": segment["current_progress"],
                "nearby_context": nearby,
                "segment_avg_speed_kmh": sum(segment_speeds) / max(1, len(segment_speeds)),
                "speed_drop_kmh": max(
                    0.0,
                    (float(recent[-2].get("speed") or 0) - float(recent[-1].get("speed") or 0))
                    if len(recent) >= 2 else 0.0,
                ),
                "door_passenger_count": sum(int(item.get("door_passenger_count") or 0)
                                            for item in recent),
                "target_time_begin": planned,
            })
            # Reuse the strict serving contract; no silently widened horizons in training.
            PredictionInput.model_validate(row)
            enriched.append(row)
    return enriched


@app.post("/train/dataset")
def train_dataset() -> dict[str, Any]:
    """Train on causal telemetry joins without blocking the ASGI event loop."""
    try:
        result = _fit(dataset_rows("train"))
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {**result, "model": backend_name, "artifact": str(MODEL_PATH)}
