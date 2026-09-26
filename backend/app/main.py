"""Schedule-only matching, bounded causal telemetry state, and dispatcher HTTP API."""
from __future__ import annotations

import asyncio
import csv
import contextlib
import logging
import math
import os
import statistics
import re
import struct
import threading
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from transit_common.contracts import (
    MAX_BATCH, BatchPredictionInput, BatchPredictionOutput, FiniteModel,
    PredictionInput, PredictionOutput, estimate_dwell_seconds, utc_time,
)
from transit_common.http import validation_error

LOG = logging.getLogger("transit.backend")
PROJECT = Path(__file__).resolve().parents[2]
ROOT = Path(os.getenv("DATASET_DIR", str(PROJECT / "dataset")))
ML_URL = os.getenv("ML_URL", "http://ml:8001")
DASHBOARD_DIR = Path(os.getenv("DASHBOARD_DIR", str(PROJECT / "dashboard")))
MAX_VEHICLES = max(1, int(os.getenv("MAX_VEHICLES", "5000")))
lock = threading.RLock()
schedule: dict[str, list[dict[str, Any]]] = {}
segment_index: dict[str, dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]]] = {}
unit_mapping: dict[str, str] = {}
vehicles: OrderedDict[str, dict[str, Any]] = OrderedDict()
history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=30))
incidents: deque[dict[str, Any]] = deque(maxlen=500)
metrics = {key: 0 for key in ("packets", "invalid", "unknown_vehicles", "predictions",
                             "ml_fallbacks", "out_of_order", "evicted", "incidents",
                             "measured_absolute_error_seconds", "measured_error_count",
                             "unsupported_ndtp_cells", "ndtp_cells_seen")}
replaying = False
schedule_generation = 0
NDTP_HOST = os.getenv("NDTP_HOST", "0.0.0.0")
NDTP_PORT = int(os.getenv("NDTP_PORT", "9201"))
NDTP_MAX_FRAME = 65535
NDTP_CELL_SIZES = {0: 26, 2: 26, 8: 6, 10: 37, 15: 50, 16: 8}
pending_predictions: dict[tuple[str, str], deque[dict[str, Any]]] = {}
current_segment_stats: dict[str, dict[str, dict[str, Any]]] = {}


class Telemetry(FiniteModel):
    """Normalized JSON telemetry; raw NDTP binary decoding needs an external adapter."""

    tr_id: str | None = Field(default=None, min_length=1, max_length=128)
    unit_id: str | None = Field(default=None, min_length=1, max_length=128)
    event_time: datetime
    lat: float | None = None
    lon: float | None = None
    speed: float | None = None
    heading: float | None = None
    location_valid: bool = True
    cur_dev_s: float | None = None
    door_open: bool | None = None
    door_passenger_count: int = Field(default=0, ge=0)

    @field_validator("event_time")
    @classmethod
    def normalize_event_time(cls, value: datetime) -> datetime:
        """Normalize naive dataset times and aware packet times to UTC."""
        return utc_time(value)


class WhatIfRequest(FiniteModel):
    """Bounded scenario inputs; route id is optional because source schedule keys by tr_id."""

    route_id: str | None = Field(default=None, min_length=1, max_length=128)
    tr_id: str | None = Field(default=None, min_length=1, max_length=128)
    extra_vehicles: int = Field(default=1, ge=1, le=10)

    @model_validator(mode="after")
    def require_scenario_key(self) -> WhatIfRequest:
        """Require a schedule key to avoid combining unrelated vehicle itineraries."""
        if not self.route_id and not self.tr_id:
            raise ValueError("Provide route_id or a scheduled tr_id")
        return self


def _dt(value: str | datetime | None) -> datetime | None:
    """Parse timestamps without silently stripping UTC offsets."""
    try:
        return utc_time(value) if value else None
    except (ValueError, TypeError):
        return None


def load_schedule(path: Path) -> int:
    """Atomically load official stops, excluding actual arrival times from feature state."""
    global schedule, segment_index, schedule_generation
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            point = re.fullmatch(r"POINT\s*\(\s*([-+\d.]+)\s+([-+\d.]+)\s*\)", row.get("geom", ""), re.I)
            time = _dt(row.get("time_begin"))
            if not point or not time or not row.get("tr_id"):
                continue
            try:
                lon, lat = float(point[1]), float(point[2])
            except ValueError:
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            grouped[row["tr_id"]].append({
                "tt_action_item_id": row["tt_action_item_id"], "time_begin": time.isoformat(),
                "building_address": row.get("building_address", ""), "route_id": row.get("route_id") or None,
                "_time": time, "_lat": lat, "_lon": lon,
            })
    if not grouped:
        raise ValueError("Schedule contains no usable official stops")
    for stops in grouped.values():
        stops.sort(key=lambda stop: stop["_time"])
        for index, stop in enumerate(stops):
            stop["_index"] = index
            stop["_count"] = len(stops)
            stop["_progress"] = index / max(1, len(stops) - 1)
            stop["_incoming_bearing"] = None
            stop["_outgoing_bearing"] = None
            stop["_segment_length_m"] = 0.0
            if index:
                previous = stops[index - 1]
                stop["_incoming_bearing"] = _bearing_deg(
                    previous["_lat"], previous["_lon"], stop["_lat"], stop["_lon"])
            if index + 1 < len(stops):
                following = stops[index + 1]
                stop["_outgoing_bearing"] = _bearing_deg(
                    stop["_lat"], stop["_lon"], following["_lat"], following["_lon"])
                stop["_segment_length_m"] = _distance_m(
                    stop["_lat"], stop["_lon"], following["_lat"], following["_lon"])
        usable_lengths = [
            stop["_segment_length_m"] for stop in stops[:-1]
            if stop["_segment_length_m"] > 1
        ]
        mean_segment_length = statistics.mean(usable_lengths) if usable_lengths else 500.0
        for stop in stops:
            stop["_mean_segment_length_m"] = mean_segment_length
    spatial: dict[str, dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]]] = {}
    # 500 m grid at Moscow latitude, with each segment inserted into every
    # cell crossed by its 500 m-expanded bounding box.
    scale_x = 111_320.0 * math.cos(math.radians(55.75))
    cell_m = 500.0
    for vehicle, stops in grouped.items():
        buckets: dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        pairs = list(zip(stops, stops[1:])) or [(stops[0], stops[0])]
        for left, right in pairs:
            x1, x2 = left["_lon"] * scale_x, right["_lon"] * scale_x
            y1, y2 = left["_lat"] * 111_320.0, right["_lat"] * 111_320.0
            for gx in range(math.floor((min(x1, x2)-cell_m)/cell_m),
                            math.floor((max(x1, x2)+cell_m)/cell_m)+1):
                for gy in range(math.floor((min(y1, y2)-cell_m)/cell_m),
                                math.floor((max(y1, y2)+cell_m)/cell_m)+1):
                    buckets[(gx, gy)].append((left, right))
        spatial[vehicle] = dict(buckets)
    with lock:
        schedule = dict(grouped)
        segment_index = spatial
        schedule_generation += 1
        vehicles.clear()
        history.clear()
        incidents.clear()
        pending_predictions.clear()
        current_segment_stats.clear()
        unit_mapping.clear()
    return sum(map(len, grouped.values()))


def _load_unit_mapping(path: Path) -> None:
    """Resolve device IDs only from unambiguous reference mappings, not geometry."""
    mapping: dict[str, set[str]] = defaultdict(set)
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("unit_id") and row.get("tr_id"):
                    mapping[row["unit_id"]].add(row["tr_id"])
    with lock:
        unit_mapping.update({key: next(iter(ids)) for key, ids in mapping.items() if len(ids) == 1})


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return clamped haversine distance, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2)**2
    return 12_742_000 * math.asin(math.sqrt(max(0, min(1, a))))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from one WGS84 point to another."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    return (math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                    math.cos(p1) * math.sin(p2) -
                                    math.sin(p1) * math.cos(p2) * math.cos(dl))) + 360) % 360


def _angle_error(left: float, right: float) -> float:
    """Return the smallest angular difference in degrees."""
    return abs((left - right + 180) % 360 - 180)


def _project_onto_segment(
    lat: float, lon: float, start: dict[str, Any], end: dict[str, Any],
) -> tuple[float, float, float]:
    """Project a GPS fix onto a scheduled stop-to-stop segment in local metres."""
    scale_y = 111_320.0
    scale_x = 111_320.0 * math.cos(math.radians((start["_lat"] + end["_lat"]) / 2))
    dx = (end["_lon"] - start["_lon"]) * scale_x
    dy = (end["_lat"] - start["_lat"]) * scale_y
    px = (lon - start["_lon"]) * scale_x
    py = (lat - start["_lat"]) * scale_y
    length_sq = dx * dx + dy * dy
    fraction = max(0.0, min(1.0, (px * dx + py * dy) / length_sq)) if length_sq else 0.0
    error = math.hypot(px - fraction * dx, py - fraction * dy)
    return fraction, error, math.sqrt(length_sq)


def match_stop(tr_id: str, lat: float | None, lon: float | None,
               moment: datetime | None = None, deviation: float = 0,
               heading: float | None = None, speed: float | None = None,
               history_points: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Map-match to the scheduled stop sequence with temporal, heading, and motion scoring.

    This scores candidate scheduled stop-to-stop segments with emission costs
    (cross-track GPS error, time, heading) and transition penalties (sequence
    progress versus elapsed time and speed). It is a one-step HMM/Viterbi-style
    matcher; the schedule coordinates are straight-line proxies, not road centerlines.
    """
    if lat is None or lon is None or moment is None:
        return None
    stops = schedule.get(tr_id, [])
    if not stops:
        return None
    valid_history = [point for point in (history_points or [])
                     if point.get("lat") is not None and point.get("lon") is not None]
    previous = valid_history[-1] if valid_history else None
    prior_candidate = None
    if previous:
        prior_id = previous.get("matched_stop_id")
        prior_candidate = next((stop for stop in stops
                                if stop["tt_action_item_id"] == prior_id), None)
        if prior_candidate is None:
            prior_candidate = min(
                stops, key=lambda stop: _distance_m(
                    previous["lat"], previous["lon"], stop["_lat"], stop["_lon"])
            )
    candidates = []
    scale_x = 111_320.0 * math.cos(math.radians(55.75))
    gx, gy = math.floor(lon*scale_x/500), math.floor(lat*111_320.0/500)
    candidates_by_key = {
        (left["_index"], right["_index"]): (left, right)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        for left, right in segment_index.get(tr_id, {}).get((gx+dx, gy+dy), [])
    }
    candidate_segments = list(candidates_by_key.values())
    if not candidate_segments and not segment_index.get(tr_id):
        candidate_segments = list(zip(stops, stops[1:])) or [(stops[0], stops[0])]
    elif not candidate_segments:
        return None
    expected_steps = 0.0
    prior_progress = 0.0
    if prior_candidate:
        # Compute transition features once per packet, not once per candidate
        # segment.  Segment means are prepared when the schedule is loaded.
        try:
            prior_progress = float(previous.get("sequence_progress"))
            if not math.isfinite(prior_progress):
                raise ValueError("non-finite prior route progress")
        except (TypeError, ValueError):
            prior_progress = float(prior_candidate["_index"])
        elapsed = max(0, (moment - utc_time(previous["event_time"])).total_seconds())
        mean_segment = float(stops[0].get("_mean_segment_length_m") or 500.0)
        expected_steps = (max(0.0, speed or 0) / 3.6) * elapsed / max(1.0, mean_segment)
    for left, right in candidate_segments:
        fraction, cross_track, segment_length = _project_onto_segment(lat, lon, left, right)
        # A schedule is a sparse polyline. Reject unlikely segments instead of
        # snapping a vehicle to a distant part of a long, sparse itinerary.
        if cross_track > 500:
            continue
        # Schedule loading normalizes these once; avoid reparsing datetimes for
        # every GPS candidate segment on every incoming packet.
        left_time = left["_time"]
        right_time = right["_time"]
        expected_time = left_time + (right_time - left_time) * fraction
        temporal_error = abs((moment - expected_time).total_seconds() - deviation)
        score = (cross_track / 35.0) ** 2 + (min(temporal_error, 10_800) / 600.0) ** 2
        expected_bearing = _bearing_deg(left["_lat"], left["_lon"], right["_lat"], right["_lon"])
        if heading is not None:
            score += _angle_error(heading, expected_bearing) / 45.0
        progress_index = left["_index"] + fraction
        if prior_candidate:
            delta = progress_index - prior_progress
            if delta < -0.75:
                score += 20 + abs(delta) * 5
            elif delta > expected_steps + 2:
                score += (delta - expected_steps - 2) * 4
        candidates.append((score, cross_track, left, right, fraction, segment_length,
                           expected_bearing, progress_index))
    if not candidates:
        return None
    score, distance, left, right, fraction, segment_length, expected_bearing, progress_index = min(
        candidates, key=lambda item: item[0])
    if score > 2500:
        return None
    nearest_stop = left if fraction <= .5 else right
    distance_to_stop = _distance_m(lat, lon, nearest_stop["_lat"], nearest_stop["_lon"])
    return {"stop_id": nearest_stop["tt_action_item_id"], "address": nearest_stop["building_address"],
            "distance_m": round(distance, 1), "planned_time": nearest_stop["time_begin"],
            "distance_to_stop_m": round(distance_to_stop, 1),
            "sequence_index": nearest_stop["_index"], "match_score": round(score, 3),
            "segment_index": left["_index"], "segment_progress": round(fraction, 4),
            "segment_distance_m": round(max(0.0, (1-fraction)*segment_length), 1),
            "sequence_progress": progress_index,
            "expected_bearing": expected_bearing,
            "heading_error_deg": round(_angle_error(heading, expected_bearing), 2)
            if heading is not None else 0.0}


def _normalize(packet: Telemetry) -> dict[str, Any]:
    """Drop invalid positions and sensor values rather than drawing or propagating them."""
    row = packet.model_dump(mode="json")
    valid = (packet.location_valid and packet.lat is not None and packet.lon is not None
             and -90 <= packet.lat <= 90 and -180 <= packet.lon <= 180)
    row["location_valid"] = valid
    if not valid:
        row["lat"] = row["lon"] = None
    if packet.speed is not None and not 0 <= packet.speed <= 180:
        row["speed"] = None
    if packet.heading is not None and not 0 <= packet.heading < 360:
        row["heading"] = None
    return row


def parse_ndtp_nav(packet: dict[str, Any]) -> Telemetry:
    """Normalize a decoded G6CellNav00 payload from the NDTP adapter.

    Longitude/latitude use the specification's absolute integer scaling and
    hemisphere bits; timestamp is Unix seconds. This endpoint intentionally
    accepts decoded JSON, while TCP framing/handshake remains an adapter concern.
    """
    nav = packet.get("G6CellNav00", packet)
    flags = int(nav.get("flags", nav.get("extraDop", 0)) or 0)
    if not flags and any(key in nav for key in ("extraDopBit5", "extraDopBit6", "extraDopBit7")):
        flags = sum(int(bool(nav.get(f"extraDopBit{bit}"))) << bit for bit in range(8))
    longitude = abs(float(nav["longitude"])) / 10_000_000
    latitude = abs(float(nav["latitude"])) / 10_000_000
    if not flags & (1 << 6):
        longitude = -longitude
    if not flags & (1 << 5):
        latitude = -latitude
    door_cells = [packet.get(name) for name in ("G6CellIrma04", "G6CellCrown03")]
    door_cells.extend(cell for cell in packet.get("cells", [])
                      if isinstance(cell, dict) and cell.get("type") in (3, 4))
    passenger_values: list[int] = []
    irma_present: dict[str, bool] = {}
    irma_closed: dict[str, bool] = {}
    for cell in door_cells:
        if not isinstance(cell, dict):
            continue
        for key, value in cell.items():
            if not isinstance(value, (int, float, bool)):
                continue
            if re.search(r"door_(in|out)\d", key, re.I):
                passenger_values.append(max(0, int(value)))
            present = re.search(r"present_door(\d+)", key, re.I)
            closed = re.search(r"closed_door(\d+)", key, re.I)
            if present:
                irma_present[present.group(1)] = bool(value)
            if closed:
                irma_closed[closed.group(1)] = bool(value)
    open_door = (any(value and not irma_closed.get(door, True)
                     for door, value in irma_present.items()) if irma_present else None)
    return Telemetry(
        tr_id=str(packet.get("tr_id")) if packet.get("tr_id") is not None else None,
        unit_id=str(packet.get("unitId", packet.get("unit_id")))
        if packet.get("unitId", packet.get("unit_id")) is not None else None,
        event_time=datetime.fromtimestamp(int(nav["timestamp"]), tz=timezone.utc),
        lat=latitude, lon=longitude,
        speed=float(nav.get("speedAvg", nav.get("speed", 0))),
        heading=float(nav.get("course", nav.get("heading", 0))),
        location_valid=bool(flags & (1 << 7)),
        door_open=open_door,
        door_passenger_count=sum(passenger_values),
    )


def decode_ndtp_nav_cell(payload: bytes, unit_id: str, tr_id: str | None = None) -> Telemetry:
    """Decode the fixed 26-byte G6CellNav00 body described by the local NDTP spec.

    The full NDTP TCP handshake/framing is still an upstream adapter responsibility.
    This decoder handles only the navigation cell body (little-endian integers).
    """
    if len(payload) != 26:
        raise ValueError("G6CellNav00 body must be exactly 26 bytes")
    timestamp = int.from_bytes(payload[0:4], "little")
    longitude = int.from_bytes(payload[4:8], "little") / 10_000_000
    latitude = int.from_bytes(payload[8:12], "little") / 10_000_000
    flags = payload[12]
    if not flags & (1 << 6):
        longitude = -longitude
    if not flags & (1 << 5):
        latitude = -latitude
    speed_avg = int.from_bytes(payload[14:16], "little")
    course = int.from_bytes(payload[18:20], "little")
    return Telemetry(
        tr_id=tr_id, unit_id=unit_id, event_time=datetime.fromtimestamp(timestamp, tz=timezone.utc),
        lat=latitude, lon=longitude, speed=float(speed_avg), heading=float(min(course, 359)),
        location_valid=bool(flags & (1 << 7)),
    )


def crc16_modbus(data: bytes) -> int:
    """Calculate the NDTP CRC-16/Modbus checksum."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc & 0xFFFF


def decode_ndtp_frame(frame: bytes) -> tuple[str, Telemetry | None]:
    """Decode one NPL/NPH frame and extract the first mandatory Nav00 cell."""
    if len(frame) < 25:
        raise ValueError("NDTP frame is shorter than NPL+NPH+Nav00")
    signature, data_size = struct.unpack_from("<HH", frame, 0)
    if signature != 0x7E7E or data_size != len(frame) - 15:
        raise ValueError("NDTP NPL signature/dataSize mismatch")
    if frame[8] != 0x02:
        raise ValueError("Unsupported NPL packet type")
    stored_crc = int.from_bytes(frame[6:8], "big")
    calculated_crc = crc16_modbus(frame[15:])
    if stored_crc != calculated_crc:
        raise ValueError("NDTP CRC-16/Modbus mismatch")
    nph = frame[15:]
    if len(nph) < 10:
        raise ValueError("Truncated NPH header")
    service_id, message_type, flags, request_id = struct.unpack_from("<HHHI", nph, 0)
    unit_id = str(struct.unpack_from("<I", frame, 9)[0])
    if message_type == 100 and service_id == 0:
        return "handshake", None
    if message_type != 101 or service_id != 1:
        raise ValueError("Unsupported NDTP NPH service/type")
    body = nph[10:]
    offset = 0
    nav: Telemetry | None = None
    can_speed: float | None = None
    while offset < len(body):
        if len(body) - offset < 2:
            raise ValueError("Truncated NDTP cell header")
        cell_type, number = body[offset], body[offset+1]
        offset += 2
        payload_size = NDTP_CELL_SIZES.get(cell_type)
        if payload_size is None:
            # Type 3/4 door cells are listed in the emulator schema without a
            # binary byte layout, so stop at that cell rather than desync framing.
            metrics["unsupported_ndtp_cells"] += 1
            break
        if len(body) - offset < payload_size:
            raise ValueError(f"Truncated NDTP cell type {cell_type}")
        payload = body[offset:offset+payload_size]
        offset += payload_size
        metrics["ndtp_cells_seen"] += 1
        if cell_type == 0:
            if number != 0 or nav is not None:
                raise ValueError("G6CellNav00 must be cell #0 and appear once")
            nav = decode_ndtp_nav_cell(payload, unit_id=unit_id)
        elif cell_type == 10:
            # G6CellCan10: four u32, fuel u16, rpm u16, engine temp i16,
            # speed u8 at byte offset 22, then five u16 axle pressures + alarm.
            can_speed = float(payload[22])
        elif cell_type == 4:
            raise ValueError("Unsupported door-cell layout")
    if nav is None:
        raise ValueError("Realtime packet lacks first G6CellNav00 cell")
    if can_speed is not None and (nav.speed is None or nav.speed == 0):
        nav = nav.model_copy(update={"speed": can_speed})
    return "realtime", nav


async def _handle_ndtp_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Receive emulator TCP frames, tolerate reconnects, and isolate invalid clients."""
    peer = writer.get_extra_info("peername")
    LOG.info("NDTP connection opened from %s", peer)
    try:
        while True:
            npl = await asyncio.wait_for(reader.readexactly(15), timeout=120)
            signature, data_size = struct.unpack_from("<HH", npl, 0)
            if signature != 0x7E7E or not 10 <= data_size <= NDTP_MAX_FRAME:
                raise ValueError("Invalid NDTP NPL signature or length")
            frame = npl + await reader.readexactly(data_size)
            packet_type, telemetry = decode_ndtp_frame(frame)
            if packet_type == "realtime" and telemetry is not None:
                result = await _process_batch([telemetry])
                if not result["results"][0].get("accepted", False):
                    LOG.debug("NDTP telemetry skipped: %s", result["results"][0])
    except asyncio.IncompleteReadError:
        LOG.info("NDTP connection closed by %s", peer)
    except asyncio.CancelledError:
        raise
    except (ValueError, OSError, asyncio.TimeoutError):
        LOG.exception("NDTP stream failed for %s", peer)
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load schedule, start NDTP TCP listener, and reuse one bounded HTTP pool."""
    candidate = ROOT / "validate" / "schedule_plan.csv"
    if not candidate.exists():
        candidate = ROOT / "train" / "schedule.csv"
    try:
        await run_in_threadpool(load_schedule, candidate)
        await run_in_threadpool(_load_unit_mapping, candidate.parent / "traffic.csv")
    except (OSError, ValueError):
        LOG.exception("Schedule unavailable; health is degraded")
    async with httpx.AsyncClient(timeout=.8, limits=httpx.Limits(max_connections=20)) as client:
        application.state.ml_client = client
        server = await asyncio.start_server(_handle_ndtp_connection, NDTP_HOST, NDTP_PORT)
        application.state.ndtp_server = server
        try:
            async with server:
                yield
        finally:
            server.close()
            await server.wait_closed()


app = FastAPI(title="Transit Delay Operations API", version="1.1.0", lifespan=lifespan)
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, check_dir=False), name="dashboard-assets")
app.add_exception_handler(RequestValidationError, validation_error)


def _fallback(payload: PredictionInput) -> PredictionOutput:
    """Distinguish missing confidence from a falsely reassuring zero risk."""
    return PredictionOutput(
        delay_seconds=payload.cur_dev_s, model="schedule-fallback", degraded=True,
        target_stop_id=payload.target_stop_id,
        horizon_minutes=(payload.target_time_begin - payload.event_time).total_seconds() / 60,
        reason="ML-сервис недоступен; вероятность задержки неизвестна",
    )


async def _predict_batch(payloads: list[PredictionInput]) -> list[PredictionOutput]:
    """Validate both sides of the wire contract before accepting any ML response."""
    try:
        response = await app.state.ml_client.post(
            f"{ML_URL}/predict/batch",
            json=BatchPredictionInput(predictions=payloads).model_dump(mode="json"),
        )
        response.raise_for_status()
        predictions = BatchPredictionOutput.model_validate(response.json()).predictions
        if len(predictions) != len(payloads):
            raise ValueError("ML response cardinality mismatch")
        for prediction, payload in zip(predictions, payloads):
            expected = (payload.target_time_begin - payload.event_time).total_seconds() / 60
            if prediction.target_stop_id != payload.target_stop_id or abs(prediction.horizon_minutes - expected) > .01:
                raise ValueError("ML response target mismatch")
        return predictions
    except (httpx.HTTPError, ValueError) as exc:
        LOG.warning("Batch inference unavailable: %s", exc)
        with lock:
            metrics["ml_fallbacks"] += len(payloads)
        return [_fallback(payload) for payload in payloads]


def _prepare(packet: Telemetry) -> tuple[dict[str, Any], tuple | None]:
    """Advance causal state under lock and freeze a request-specific feature snapshot."""
    row = _normalize(packet)
    with lock:
        metrics["packets"] += 1
        if not row["location_valid"]:
            metrics["invalid"] += 1
        tr_id = packet.tr_id or unit_mapping.get(packet.unit_id or "")
        key = tr_id or f"unit:{packet.unit_id}"
        row["tr_id"] = tr_id
        previous = vehicles.get(key)
        if previous and packet.event_time <= utc_time(previous["event_time"]):
            metrics["out_of_order"] += 1
            return {"accepted": False, "reason": "duplicate_or_out_of_order", "forecast": None}, None
        if previous and (packet.event_time - utc_time(previous["event_time"])).total_seconds() > 300:
            history.pop(key, None)
            for pending_key in [candidate for candidate in pending_predictions if candidate[0] == key]:
                pending_predictions.pop(pending_key, None)
            current_segment_stats.pop(key, None)
            previous = None
        scheduled = key in schedule
        row.update({"scheduled": scheduled, "forecast": None})
        # Keep the last known position for the dashboard when a telemetry
        # packet has an invalid/missing GPS fix.  The fix remains explicitly
        # marked stale and is not fed back into map matching below.
        position_stale = not row["location_valid"]
        if position_stale and previous and previous.get("location_valid"):
            row["lat"] = previous.get("lat")
            row["lon"] = previous.get("lon")
            row["location_valid"] = True
            row["position_stale"] = True
        else:
            row["position_stale"] = False
        vehicles[key] = row
        vehicles.move_to_end(key)
        while len(vehicles) > MAX_VEHICLES:
            evicted, _ = vehicles.popitem(last=False)
            history.pop(evicted, None)
            metrics["evicted"] += 1
        if not scheduled:
            metrics["unknown_vehicles"] += 1
            return {"accepted": True, "scheduled": False, "forecast": None}, None
        if row["cur_dev_s"] is None and previous:
            row["cur_dev_s"] = previous.get("cur_dev_s")
        match = match_stop(key, None if position_stale else row["lat"],
                           None if position_stale else row["lon"],
                           packet.event_time, row["cur_dev_s"] or 0,
                           row["heading"], row["speed"], list(history[key]))
        if (match and packet.cur_dev_s is None and match["distance_to_stop_m"] <= 60
                and (row["speed"] or 0) <= 3):
            pending_key = (key, match["stop_id"])
            pending_items = pending_predictions.pop(pending_key, deque())
            if pending_items:
                actual_delay = (packet.event_time - utc_time(match["planned_time"])).total_seconds()
                errors = [abs(actual_delay - float(item["predicted_delay_seconds"]))
                          for item in pending_items]
                with lock:
                    metrics["measured_absolute_error_seconds"] += sum(errors)
                    metrics["measured_error_count"] += len(errors)
                incidents.append({
                    "tr_id": key, "target_stop_id": match["stop_id"],
                    "actual_delay_seconds": round(actual_delay, 2),
                    "absolute_error_seconds": round(errors[-1], 2),
                    "mean_absolute_error_seconds": round(sum(errors) / len(errors), 2),
                    "measured_predictions": len(errors),
                    "measured_at": packet.event_time.isoformat(),
                    "position": {"lat": row["lat"], "lon": row["lon"]},
                })
                metrics["incidents"] += 1
        row["matched_stop_id"] = match["stop_id"] if match else None
        row["sequence_progress"] = (match["sequence_progress"]
                                    if match else None)
        if (packet.cur_dev_s is None and match and match["distance_to_stop_m"] <= 60
                and row["speed"] is not None and row["speed"] <= 3
                and (not previous or (previous.get("matched_stop") or {}).get("stop_id") != match["stop_id"])):
            row["cur_dev_s"] = (packet.event_time - utc_time(match["planned_time"])).total_seconds()
        row["matched_stop"] = match
        segment_key = match["segment_index"] if match else None
        segment_state = current_segment_stats.get(key)
        if (previous is None or segment_state is None
                or segment_state.get("segment_index") != segment_key):
            segment_state = {"segment_index": segment_key, "speeds": deque(maxlen=30)}
            current_segment_stats[key] = segment_state
        if row.get("speed") is not None:
            segment_state["speeds"].append(float(row["speed"]))
        row["segment_avg_speed_kmh"] = (
            sum(segment_state["speeds"]) / len(segment_state["speeds"])
            if segment_state["speeds"] else 0.0
        )
        row["segment_speed_samples"] = len(segment_state["speeds"])
        row["speed_drop_kmh"] = max(
            0.0, float(previous.get("speed") or 0) - float(row["speed"] or 0)
        ) if previous else 0.0
        row["door_passenger_delta"] = max(
            0, packet.door_passenger_count - int(previous.get("door_passenger_count") or 0)
        ) if previous else packet.door_passenger_count
        while history[key] and (packet.event_time - utc_time(history[key][0]["event_time"])).total_seconds() > 300:
            history[key].popleft()
        history[key].append({field: row.get(field) for field in
                              ("event_time", "speed", "lat", "lon", "heading", "door_open")}
                            | {"matched_stop_id": row["matched_stop_id"],
                               "sequence_progress": row["sequence_progress"],
                               "matched_segment_index": segment_key,
                               "door_passenger_count": row["door_passenger_delta"]})
        result = {"accepted": True, "scheduled": True, "matched_stop": match, "forecast": None}
        route = schedule[key]
        lower_bound = packet.event_time + timedelta(seconds=600)
        upper_bound = packet.event_time + timedelta(seconds=900)
        # Stops are sorted during schedule loading. Locate the first stop
        # strictly after T+600s in O(log n), rather than scanning the whole
        # itinerary for every telemetry packet.
        low, high = 0, len(route)
        while low < high:
            middle = (low + high) // 2
            if route[middle]["_time"] <= lower_bound:
                low = middle + 1
            else:
                high = middle
        target = route[low] if low < len(route) and route[low]["_time"] <= upper_bound else None
        if target is None:
            return result, None
        context = sum(
            1 for other in vehicles.values()
            if not other["scheduled"] and other["location_valid"] and row["location_valid"]
            and 0 <= (packet.event_time - utc_time(other["event_time"])).total_seconds() <= 120
            and _distance_m(row["lat"], row["lon"], other["lat"], other["lon"]) <= 500
        )
        payload = PredictionInput(
            tr_id=key, event_time=packet.event_time, target_time_begin=target["_time"],
            target_stop_id=target["tt_action_item_id"], cur_dev_s=row["cur_dev_s"],
            speed=row["speed"] or 0, lat=row["lat"], lon=row["lon"], heading=row["heading"],
            stop_distance_m=_distance_m(row["lat"], row["lon"], target["_lat"], target["_lon"])
            if row["location_valid"] else 0,
            nearby_context=context, history=list(history[key]),
            target_stop_lat=target["_lat"], target_stop_lon=target["_lon"],
            target_stop_index=target["_index"], route_stop_count=target["_count"],
            target_progress=target["_progress"],
            current_progress=match["sequence_progress"] / max(1, len(schedule[key])-1) if match else 0,
            segment_index=match["segment_index"] if match else 0,
            segment_progress=match["segment_progress"] if match else 0,
            segment_distance_m=match["segment_distance_m"] if match else 0,
            heading_error_deg=match["heading_error_deg"] if match else 0,
            door_open_events=sum(1 for point in history[key] if point.get("door_open")),
            door_passenger_count=sum(
                int(point.get("door_passenger_count") or 0)
                for point in list(history[key])[-5:]
            ),
            stopped_seconds=estimate_dwell_seconds(list(history[key])),
            segment_avg_speed_kmh=row["segment_avg_speed_kmh"],
            speed_drop_kmh=row["speed_drop_kmh"],
        )
        return result, (key, row, payload, schedule_generation)


def _store_prediction(queued: tuple, output: PredictionOutput) -> dict[str, Any]:
    """Prevent slow earlier responses from overwriting newer vehicle states."""
    key, row, _, generation = queued
    forecast = output.model_dump()
    probability = output.delay_probability
    risk = ("unknown" if output.degraded or probability is None else "high" if probability >= .7
            else "medium" if probability >= .35 else "low")
    payload = queued[2]
    matched_stop = row.get("matched_stop") or {}
    forecast.update({
        "delay_minutes": None if output.delay_seconds is None else round(output.delay_seconds / 60, 1),
        "risk": risk,
        "reason": output.reason or "Предупреждающий паттерн не классифицирован",
        "recommendation": "Проверить телеметрию и обстановку; рекомендация не заменяет решение диспетчера.",
        "problem_segment": {
            "stop_id": row.get("matched_stop_id"),
            "segment_index": matched_stop.get("segment_index"),
            "segment_progress": matched_stop.get("segment_progress"),
            "distance_to_next_stop_m": matched_stop.get("segment_distance_m"),
        },
        "pattern_evidence": {
            "speed_drop_kmh": payload.speed_drop_kmh,
            "segment_avg_speed_kmh": payload.segment_avg_speed_kmh,
            "stopped_seconds": payload.stopped_seconds,
            "door_events": payload.door_open_events,
            "door_passengers": payload.door_passenger_count,
        },
    })
    pattern = _classify_warning_pattern(payload, row)
    forecast["pattern_code"] = pattern["code"]
    if not (output.degraded and output.reason):
        forecast["reason"] = pattern["reason"]
    forecast["recommendation"] = pattern["recommendation"]
    forecast["pattern_evidence"].update({
        "current_speed_kmh": row.get("speed"),
        "segment_speed_samples": row.get("segment_speed_samples", 0),
        "cur_dev_s": payload.cur_dev_s,
        "nearby_context": payload.nearby_context,
        "heading_error_deg": payload.heading_error_deg,
    })
    with lock:
        metrics["predictions"] += 1
        if generation != schedule_generation or vehicles.get(key) is not row:
            return forecast
        row["forecast"] = forecast
        if output.delay_seconds is not None:
            pending_key = (key, output.target_stop_id)
            prediction_queue = pending_predictions.setdefault(pending_key, deque(maxlen=120))
            prediction_queue.append({
                "target_stop_id": output.target_stop_id,
                "predicted_delay_seconds": output.delay_seconds,
                "updated_at": row["event_time"],
            })
            while len(pending_predictions) > MAX_VEHICLES * 2:
                pending_predictions.pop(next(iter(pending_predictions)))
            forecast["absolute_error_seconds"] = None
        if risk in {"high", "medium"}:
            incidents.append({"tr_id": key, **forecast, "updated_at": row["event_time"],
                              "position": {"lat": row["lat"], "lon": row["lon"]}})
    return forecast


def _classify_warning_pattern(payload: PredictionInput, row: dict[str, Any]) -> dict[str, str]:
    """Name the strongest observed warning signal without asserting its cause.

    This is a small, transparent backend rule set for dispatcher explanations,
    not a replacement for the ML model's risk score or a causal diagnosis.
    """
    speed = float(row.get("speed") or 0)
    matched = row.get("matched_stop") or {}
    distance_to_stop = float(matched.get("distance_to_stop_m") or 0)
    segment_samples = int(row.get("segment_speed_samples") or 0)
    segment_mean = float(payload.segment_avg_speed_kmh or 0)

    if payload.speed_drop_kmh >= 15 and speed < 15:
        return {
            "code": "sharp_speed_drop",
            "reason": "Наблюдается резкое снижение скорости на сегменте",
            "recommendation": "Проверить обстановку на участке; возможны затор или препятствие.",
        }
    if payload.stopped_seconds >= 60 and (
        payload.door_open_events > 0 or payload.door_passenger_count > 0
    ):
        return {
            "code": "extended_boarding_stop",
            "reason": "Длительная остановка с активностью дверей или пассажирским обменом",
            "recommendation": "Проверить длительность посадки/высадки и работу остановки.",
        }
    if (payload.stopped_seconds >= 90 and speed <= 1
            and matched and distance_to_stop <= 80):
        return {
            "code": "long_stop_near_scheduled_stop",
            "reason": "ТС долго стоит рядом с остановкой расписания",
            "recommendation": "Проверить фактическую остановку, посадку и время отправления.",
        }
    if (segment_samples >= 3 and 0 < segment_mean <= 10 and speed <= 10
            and payload.segment_distance_m > 100):
        return {
            "code": "sustained_low_segment_speed",
            "reason": "Устойчивая низкая скорость на текущем сегменте",
            "recommendation": "Проверить ситуацию на участке; сигнал основан на скоростях последних телеметрических точек.",
        }
    if payload.cur_dev_s is not None and payload.cur_dev_s >= 120:
        return {
            "code": "observed_schedule_lateness",
            "reason": "Уже зафиксировано отставание от расписания",
            "recommendation": "Проверить величину отставания и рассмотреть диспетчерские меры.",
        }
    if payload.nearby_context >= 4:
        return {
            "code": "nearby_vehicle_density",
            "reason": "Рядом наблюдается повышенное число контекстных ТС",
            "recommendation": "Проверить дорожную обстановку; соседние ТС сами по себе не подтверждают затор.",
        }
    if speed >= 5 and payload.heading_error_deg >= 75:
        return {
            "code": "heading_route_mismatch",
            "reason": "Курс ТС заметно расходится с направлением сопоставленного сегмента",
            "recommendation": "Проверить качество GPS и результат сопоставления с расписанием.",
        }
    return {
        "code": "no_single_pattern",
        "reason": "Отдельный предупреждающий сигнал не выделен; оценка риска учитывает совокупность признаков",
        "recommendation": "Проверить телеметрию и прогноз в динамике; это не установленная причина задержки.",
    }


async def _process_batch(packets: list[Telemetry]) -> dict[str, Any]:
    """Use one processing path for single, batch, and historical telemetry."""
    results: list[dict[str, Any]] = [{} for _ in packets]
    queued = []
    for index in sorted(range(len(packets)), key=lambda i: packets[i].event_time):
        try:
            result, item = _prepare(packets[index])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            # One malformed/edge-case packet must not abort a whole historical
            # replay batch or the live stream.  Keep the error observable while
            # allowing the remaining vehicles to reach the dashboard.
            LOG.warning("Skipping packet %s during processing: %s",
                        packets[index].tr_id or packets[index].unit_id, exc)
            with lock:
                metrics["invalid"] += 1
            result, item = {
                "accepted": False, "reason": "processing_error", "forecast": None,
            }, None
        results[index] = result
        if item:
            queued.append((index, item))
    if queued:
        predictions = await _predict_batch([item[2] for _, item in queued])
        for (index, item), prediction in zip(queued, predictions):
            results[index]["forecast"] = _store_prediction(item, prediction)
    return {"processed": len(packets), "results": results}


@app.get("/health")
async def health() -> dict[str, Any]:
    """Expose missing schedules as degraded, not unconditional success."""
    return {"status": "ok" if schedule else "degraded", "scheduled_vehicles": len(schedule),
            "stops": sum(map(len, schedule.values())), "metrics": dict(metrics),
            "online_mae_seconds": (
                metrics["measured_absolute_error_seconds"] / metrics["measured_error_count"]
                if metrics["measured_error_count"] else None
            ),
            "ndtp_tcp": {"host": NDTP_HOST, "port": NDTP_PORT, "active": bool(
                getattr(app.state, "ndtp_server", None))},
            "replaying": replaying}


@app.get("/api/schedule")
async def get_schedule() -> dict[str, Any]:
    """Return official vehicle itineraries, not invented route identifiers."""
    return {"routes": [{"tr_id": key, "stops": [
        {"stop_id": stop["tt_action_item_id"], "time": stop["time_begin"],
         "lat": stop["_lat"], "lon": stop["_lon"], "address": stop["building_address"]}
        for stop in stops]} for key, stops in schedule.items()]}


@app.post("/api/telemetry")
async def ingest(packet: Telemetry) -> dict[str, Any]:
    """Accept a normalized packet; unknown devices are context only."""
    if replaying:
        raise HTTPException(409, "Historical replay is active")
    return (await _process_batch([packet]))["results"][0]


@app.post("/api/telemetry/ndtp/nav")
async def ingest_ndtp_nav(packet: dict[str, Any]) -> dict[str, Any]:
    """Accept a decoded NDTP G6CellNav00 JSON cell from an external TCP adapter."""
    if replaying:
        raise HTTPException(409, "Historical replay is active")
    try:
        normalized = parse_ndtp_nav(packet)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(422, f"Invalid G6CellNav00 payload: {exc}") from exc
    return (await _process_batch([normalized]))["results"][0]


@app.post("/api/telemetry/ndtp/nav-cell/{unit_id}")
async def ingest_ndtp_nav_cell(
    unit_id: str, request: Request, tr_id: str | None = None,
) -> dict[str, Any]:
    """Decode a raw 26-byte G6CellNav00 body from a TCP framing adapter."""
    if replaying:
        raise HTTPException(409, "Historical replay is active")
    try:
        normalized = decode_ndtp_nav_cell(await request.body(), unit_id=unit_id, tr_id=tr_id)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(422, f"Invalid G6CellNav00 body: {exc}") from exc
    return (await _process_batch([normalized]))["results"][0]


@app.post("/api/telemetry/batch")
async def ingest_batch(packets: Annotated[list[Any], Body(max_length=MAX_BATCH)]) -> dict[str, Any]:
    """Isolate invalid rows, preserving input order and limiting request size."""
    if replaying:
        raise HTTPException(409, "Historical replay is active")
    results: list[dict[str, Any]] = [{} for _ in packets]
    valid, indices = [], []
    for index, packet in enumerate(packets):
        try:
            valid.append(Telemetry.model_validate(packet))
            indices.append(index)
        except ValidationError:
            results[index] = {"accepted": False, "reason": "invalid_packet", "forecast": None}
            metrics["invalid"] += 1
    processed = await _process_batch(valid)
    for index, result in zip(indices, processed["results"]):
        results[index] = result
    return {"processed": len(packets), "results": results}


def _read_replay(path: Path) -> tuple[list[Telemetry], int]:
    """Read/sort historical CSV in a worker; source files are not chronological."""
    packets, rejected = [], 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            try:
                packets.append(Telemetry(
                    tr_id=raw.get("tr_id") or None, unit_id=raw.get("unit_id") or None,
                    event_time=raw["event_time"],
                    **{key: float(raw[key]) if raw.get(key) else None for key in ("lat", "lon", "speed", "heading")},
                    location_valid=raw.get("location_valid", "").lower() == "true",
                ))
            except (ValueError, KeyError):
                rejected += 1
    packets.sort(key=lambda item: item.event_time)
    return packets, rejected


@app.post("/api/replay/{split}")
async def replay(split: str) -> dict[str, Any]:
    """Replay a split with its own schedule in time order, excluding live writes."""
    global replaying
    if split not in {"train", "test", "validate"}:
        raise HTTPException(400, "split must be train, test, or validate")
    path = ROOT / split
    reference = path / ("schedule_plan.csv" if split == "validate" else "schedule.csv")
    if not reference.exists() or not (path / "traffic.csv").exists():
        raise HTTPException(404, "Split files unavailable")
    if replaying:
        raise HTTPException(409, "Historical replay already active")
    replaying = True
    try:
        packets, rejected = await run_in_threadpool(_read_replay, path / "traffic.csv")
        await run_in_threadpool(load_schedule, reference)
        await run_in_threadpool(_load_unit_mapping, path / "traffic.csv")
        accepted = 0
        for start in range(0, len(packets), MAX_BATCH):
            result = await _process_batch(packets[start:start + MAX_BATCH])
            accepted += sum(item["accepted"] for item in result["results"])
            await asyncio.sleep(0)
        return {"split": split, "ingested": accepted, "rejected": rejected,
                "skipped": len(packets) - accepted}
    finally:
        replaying = False


@app.get("/api/state")
async def state() -> dict[str, Any]:
    """Provide bounded latest state and incident history."""
    with lock:
        return {"vehicles": list(vehicles.values()), "incidents": list(incidents), "metrics": dict(metrics)}


@app.post("/api/what-if")
async def what_if(request: WhatIfRequest) -> dict[str, Any]:
    """Estimate an assumption-bound headway scenario from one official vehicle itinerary."""
    if request.tr_id and request.tr_id not in schedule:
        raise HTTPException(404, "tr_id is not present in the official schedule")
    if request.route_id:
        route_vehicles = {vehicle for vehicle, vehicle_stops in schedule.items()
                          if any(stop.get("route_id") == request.route_id for stop in vehicle_stops)}
        if not route_vehicles:
            raise HTTPException(404, "route_id is not present in the official schedule")
        if len(route_vehicles) != 1:
            raise HTTPException(422, "Select a tr_id; multiple vehicle itineraries are not a route network")
        itinerary_id = next(iter(route_vehicles))
    else:
        itinerary_id = request.tr_id
    stop_rows = schedule.get(itinerary_id, [])
    if not stop_rows:
        raise HTTPException(404, "No official schedule is loaded")
    times = sorted(stop["_time"] for stop in stop_rows)
    span = (times[-1] - times[0]).total_seconds() / 60 if len(times) > 1 else 0.0
    positive_gaps = [(right-left).total_seconds()/60 for left, right in zip(times, times[1:]) if right > left]
    typical_gap = statistics.median(positive_gaps) if positive_gaps else 0.0
    # Source data has one trip itinerary, not a route/fleet timetable. This is
    # only a hypothetical evenly-spaced schedule approximation.
    before = typical_gap
    projected_vehicles = request.extra_vehicles + 1
    after = before / projected_vehicles
    return {
        "route_id": request.route_id, "tr_id": itinerary_id,
        "additional_vehicles": request.extra_vehicles, "scheduled_stops": len(stop_rows),
        "observed_scheduled_vehicles": int(itinerary_id in vehicles),
        "projected_vehicles": projected_vehicles,
        "assumed_span_minutes": round(span, 2),
        "scheduled_gap_median_minutes": round(typical_gap, 2),
        "estimated_headway_before_minutes": round(before, 2),
        "estimated_headway_after_minutes": round(after, 2),
        "estimated_headway_reduction_minutes": round(max(0, before-after), 2),
        "assumptions": [
            "Используется одно расписание tr_id; route_id и fleet timetable в данных отсутствуют.",
            "Используются только официальные остановки из загруженного расписания.",
            "Базовый интервал принят равным медиане временных промежутков между остановками этого ТС.",
            "Дополнительный выпуск условно равномерно делит этот интервал на число гипотетических ТС.",
            "Не моделируются оборотное время, дорожные условия, вместимость и влияние на другие рейсы; это не рекомендация к фактическому выпуску.",
        ],
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the dashboard in both workspace and Docker layouts."""
    page = (DASHBOARD_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(page, headers={"Cache-Control": "no-store, max-age=0"})
