"""Train on labeled data and create the semicolon-delimited validation submission."""
from __future__ import annotations

import argparse
import bisect
import csv
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_service.app import main as ml  # noqa: E402
from transit_common.contracts import PredictionInput, utc_time  # noqa: E402

LOG = logging.getLogger("transit.submission")


def _read_telemetry(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load finite, valid positions and sensor values grouped by scheduled vehicle."""
    by_vehicle: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                moment = utc_time(row["event_time"])
            except (KeyError, TypeError, ValueError):
                continue
            packet: dict[str, Any] = {"_time": moment}
            for field, lower, upper in (("speed", 0, 180), ("heading", 0, 359.99999),
                                        ("lat", -90, 90), ("lon", -180, 180)):
                try:
                    number = float(row[field])
                    packet[field] = number if lower <= number <= upper else None
                except (KeyError, TypeError, ValueError):
                    packet[field] = None
            if str(row.get("location_valid", "")).lower() != "true":
                packet["lat"] = packet["lon"] = None
            by_vehicle.setdefault(row.get("tr_id", ""), []).append(packet)
    for rows in by_vehicle.values():
        rows.sort(key=lambda item: item["_time"])
    return by_vehicle


def generate(output: Path, retrain: bool = True, service_url: str | None = None) -> dict[str, Any]:
    """Train/load the fitted project model and generate one prediction per validation sample."""
    validate = ROOT / "dataset" / "validate"
    client = None
    if service_url:
        import httpx
        client = httpx.Client(base_url=service_url.rstrip("/"), timeout=120)
        if retrain:
            response = client.post("/train/dataset")
            response.raise_for_status()
            report = response.json()
            LOG.info("Service trained %s; holdout MAE %.2f sec, baseline %.2f sec",
                     report["model"], report["validation_mae_seconds"], report["baseline_mae_seconds"])
        health = client.get("/health")
        health.raise_for_status()
        model_name = health.json()["model"]
        if model_name == "cur_dev_baseline" or health.json()["status"] == "degraded":
            raise RuntimeError("ML service has no trained project model; refusing baseline submission")
    else:
        if retrain:
            report = ml._fit(ml.dataset_rows())
            LOG.info("Trained %s; holdout MAE %.2f sec, baseline %.2f sec",
                     ml.backend_name, report["validation_mae_seconds"], report["baseline_mae_seconds"])
        elif ml.model is None:
            ml.startup()
            if ml.model is None:
                raise RuntimeError("No compatible saved model; run again without --reuse-model")
        if ml.model is None or ml.backend_name == "cur_dev_baseline":
            raise RuntimeError("No trained project model is active; refusing to create a baseline submission")

    schedule: dict[tuple[str, str], dict[str, Any]] = {}
    route_stops: dict[str, list[dict[str, Any]]] = {}
    with (validate / "schedule_plan.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                lon, lat = map(float, row["geom"].removeprefix("POINT (").removesuffix(")").split())
            except (ValueError, AttributeError):
                continue
            stop = {"id": row["tt_action_item_id"], "time": utc_time(row["time_begin"]),
                    "lat": lat, "lon": lon}
            schedule[(row["tr_id"], row["tt_action_item_id"])] = stop
            route_stops.setdefault(row["tr_id"], []).append(stop)
    for stops in route_stops.values():
        stops.sort(key=lambda item: item["time"])
        for index, stop in enumerate(stops):
            stop["index"] = index
            stop["count"] = len(stops)
    # Use the same bounded spatial candidate index as dataset_rows during fit.
    # Besides avoiding an O(points * stops) full scan, this ensures that the
    # identical map-matching path and official segment indices reach inference.
    schedule_segments = ml._build_schedule_segment_index(route_stops)
    telemetry = _read_telemetry(validate / "traffic.csv")
    times = {vehicle: [row["_time"] for row in rows] for vehicle, rows in telemetry.items()}
    requests: list[tuple[str, PredictionInput]] = []
    with (validate / "points.csv").open(encoding="utf-8-sig", newline="") as stream:
        for point in csv.DictReader(stream):
            vehicle, stop_id = point["tr_id"], point["target_stop_id"]
            target_stop = schedule.get((vehicle, stop_id))
            planned = target_stop["time"] if target_stop else None
            if planned is None or planned != utc_time(point["target_time_begin"]):
                raise ValueError(f"{point['sample_id']}: target stop/time is not in official schedule")
            t = utc_time(point["T"])
            horizon = (planned - t).total_seconds()
            if not 600 < horizon <= 900:
                raise ValueError(f"{point['sample_id']}: target outside strict 10–15 minute horizon")
            entries = telemetry.get(vehicle, [])
            index = bisect.bisect_right(times.get(vehicle, []), t) - 1
            current = entries[index] if index >= 0 and (t-entries[index]["_time"]).total_seconds() <= 300 else {}
            history = [
                {"event_time": row["_time"], "speed": row["speed"], "lat": row["lat"],
                 "lon": row["lon"], "heading": row["heading"], "door_open": row.get("door_open")}
                for row in entries[max(0, index-29):index+1]
                if index >= 0 and (t-row["_time"]).total_seconds() <= 300
            ]
            deviation = float(point["cur_dev_s"] or 0)
            match_features = ml._map_match_features(
                route_stops[vehicle], current.get("lat"), current.get("lon"),
                t, current.get("heading"),
                spatial_index=schedule_segments[vehicle],
            )
            current_progress = (
                float(match_features["current_progress"])
                if match_features else 0.0
            )
            enriched_history = []
            for sample in history:
                sample_match = ml._map_match_features(
                    route_stops[vehicle], sample.get("lat"), sample.get("lon"),
                    sample["event_time"], sample.get("heading"),
                    spatial_index=schedule_segments[vehicle],
                )
                enriched_history.append({**sample,
                                         "matched_segment_index": sample_match["segment_index"]})
            history = enriched_history
            recent_speeds = [float(item.get("speed") or 0) for item in history]
            segment_speeds = [float(item.get("speed") or 0) for item in history
                              if item.get("matched_segment_index") == match_features["segment_index"]]
            request = PredictionInput(
                tr_id=vehicle, event_time=t, target_time_begin=planned,
                target_stop_id=stop_id, cur_dev_s=deviation,
                speed=current.get("speed") or 0, lat=current.get("lat"), lon=current.get("lon"),
                heading=current.get("heading"), history=history,
                target_stop_lat=target_stop["lat"], target_stop_lon=target_stop["lon"],
                target_stop_index=target_stop["index"], route_stop_count=target_stop["count"],
                target_progress=target_stop["index"] / max(1, target_stop["count"] - 1),
                stop_distance_m=ml._distance_m_safe(
                    current.get("lat"), current.get("lon"), target_stop["lat"], target_stop["lon"]),
                current_progress=current_progress,
                segment_index=match_features["segment_index"],
                segment_progress=match_features["segment_progress"],
                segment_distance_m=match_features["segment_distance_m"],
                heading_error_deg=match_features["heading_error_deg"],
                door_open_events=sum(bool(point.get("door_open")) for point in history),
                stopped_seconds=ml._stationary_seconds(history),
                door_passenger_count=sum(int(item.get("door_passenger_count") or 0)
                                         for item in history[-5:]),
                segment_avg_speed_kmh=sum(segment_speeds) / max(1, len(segment_speeds)),
                speed_drop_kmh=max(
                    0.0, recent_speeds[-2] - recent_speeds[-1]
                    if len(recent_speeds) >= 2 else 0.0,
                ),
            )
            requests.append((point["sample_id"], request))

    submission: list[tuple[str, float]] = []
    for start in range(0, len(requests), 64):
        chunk = requests[start:start + 64]
        if client:
            from transit_common.contracts import BatchPredictionInput, BatchPredictionOutput
            response = client.post(
                "/predict/batch",
                json=BatchPredictionInput(predictions=[request for _, request in chunk]).model_dump(mode="json"),
            )
            response.raise_for_status()
            predictions = BatchPredictionOutput.model_validate(response.json()).predictions
        else:
            from transit_common.contracts import BatchPredictionInput
            predictions = ml.predict_batch(BatchPredictionInput(
                predictions=[request for _, request in chunk]
            )).predictions
        if len(predictions) != len(chunk):
            raise RuntimeError("Model returned a wrong number of predictions")
        for (sample_id, _), prediction in zip(chunk, predictions):
            if prediction.delay_seconds is None or prediction.degraded:
                raise RuntimeError(f"{sample_id}: trained model did not return a valid prediction")
            if prediction.model == "cur_dev_baseline":
                raise RuntimeError(f"{sample_id}: baseline is not an accepted model prediction")
            if not math.isfinite(float(prediction.delay_seconds)):
                raise RuntimeError(f"{sample_id}: model returned a non-finite delay")
            submission.append((sample_id, float(prediction.delay_seconds)))
    if client:
        client.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter=";", lineterminator="\n")
            writer.writerow(["sample_id", "prediction"])
            writer.writerows(submission)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"predictions": len(submission), "scheduled_vehicles": len(route_stops),
            "unknown_telemetry_vehicles": len(set(telemetry) - set(route_stops)),
            "model": model_name if service_url else ml.backend_name}


def main() -> None:
    """Parse the command line and report the generated CSV path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--reuse-model", action="store_true",
                        help="Use the compatible artifact under models/ instead of retraining.")
    parser.add_argument("--service-url",
                        help="Use the running ML Docker service, e.g. http://127.0.0.1:8001.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = generate(args.output.resolve(), retrain=not args.reuse_model,
                       service_url=args.service_url)
    print(f"Модель: {summary['model']}")
    print(f"Создан {args.output.resolve()}: {summary['predictions']} прогнозов; "
          f"расписных ТС: {summary['scheduled_vehicles']}; "
          f"контекстных ТС: {summary['unknown_telemetry_vehicles']}")


if __name__ == "__main__":
    main()
