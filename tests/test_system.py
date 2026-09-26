"""Regression coverage for causal ingestion, strict contracts, and service integration."""
from __future__ import annotations

import asyncio
import csv
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import main as backend
from ml_service.app import main as ml
from transit_common.contracts import BatchPredictionInput, PredictionInput, PredictionOutput, utc_time

ROOT = Path(__file__).resolve().parents[1]
NOW = utc_time("2026-01-06T12:00:00Z")


def request_data(**changes):
    """Build a valid strict-horizon request."""
    result = {"tr_id": "bus", "target_stop_id": "future", "event_time": NOW.isoformat(),
              "target_time_begin": (NOW + timedelta(minutes=12)).isoformat(), "cur_dev_s": 240}
    result.update(changes)
    return result


def telemetry(**changes):
    """Build a valid telemetry dictionary."""
    result = {"tr_id": "bus", "event_time": NOW.isoformat(), "lat": 55.7, "lon": 37.6,
              "speed": 10, "cur_dev_s": 240}
    result.update(changes)
    return result


def stop(identity, moment):
    """Create an official stop fixture."""
    return {"tt_action_item_id": identity, "_time": moment, "_index": 0, "_count": 2,
            "_progress": 0.0, "time_begin": moment.isoformat(),
            "_lat": 55.7, "_lon": 37.6, "building_address": "Address"}


def reset_state():
    """Ensure test cases do not depend on global state left by earlier tests."""
    backend.schedule = {"bus": [stop("past", NOW - timedelta(minutes=4)),
                                 stop("future", NOW + timedelta(minutes=12))]}
    backend.vehicles.clear()
    backend.history.clear()
    backend.incidents.clear()
    backend.unit_mapping.clear()
    backend.replaying = False
    for key in backend.metrics:
        backend.metrics[key] = 0
    ml.model = None


class ContractsTests(unittest.TestCase):
    """Validate temporal and numerical boundaries through real HTTP validation."""

    def setUp(self):
        """Start each test with a clean baseline."""
        reset_state()
        self.client = TestClient(ml.app)

    def test_strict_horizon_boundaries(self):
        """Only (600,900] seconds are accepted."""
        for seconds, expected in [(599, 422), (600, 422), (601, 200), (900, 200), (901, 422)]:
            response = self.client.post("/predict", json=request_data(
                target_time_begin=(NOW + timedelta(seconds=seconds)).isoformat()))
            self.assertEqual(response.status_code, expected, response.text)

    def test_missing_or_bad_dates_cannot_bypass_horizon(self):
        """Missing and malformed dates must not silently become a 12.5-minute prediction."""
        for key in ("event_time", "target_time_begin", "target_stop_id"):
            payload = request_data()
            del payload[key]
            self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)
        self.assertEqual(self.client.post("/predict", json=request_data(event_time="bad")).status_code, 422)

    def test_timezone_preserves_instant(self):
        """UTC and Moscow offsets describe the same prediction horizon."""
        payload = request_data(event_time="2026-01-06T15:00:00+03:00")
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["horizon_minutes"], 12)

    def test_future_history_rejected(self):
        """Feature history may not look beyond T."""
        response = self.client.post("/predict", json=request_data(
            history=[{"event_time": (NOW + timedelta(seconds=1)).isoformat(), "speed": 20}]))
        self.assertEqual(response.status_code, 422)

    def test_finite_values(self):
        """NaN and infinity are rejected before inference or response serialization."""
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValidationError):
                PredictionInput(**request_data(cur_dev_s=value))
            with self.assertRaises(ValidationError):
                backend.Telemetry(**telemetry(lat=value))
        response = self.client.post("/predict", content=json.dumps(request_data(cur_dev_s=float("nan"))),
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    def test_missing_model_is_not_low_risk(self):
        """An untrained baseline never advertises a fake calibrated probability or MAE."""
        response = self.client.post("/predict", json=request_data())
        result = response.json()
        self.assertEqual(result["delay_seconds"], 240)
        self.assertIsNone(result["delay_probability"])
        self.assertTrue(result["degraded"])
        self.assertNotIn("predicted_absolute_error_seconds", result)

    def test_batch_is_bounded_and_typed(self):
        """Validate the actual JSON envelope and limit."""
        self.assertEqual(self.client.post("/predict/batch", json=[request_data()]).status_code, 422)
        self.assertEqual(self.client.post("/predict/batch", json={"predictions": []}).status_code, 422)
        self.assertEqual(self.client.post("/predict/batch", json={"predictions": [request_data()] * 257}).status_code, 422)

    def test_feature_speed_delta(self):
        """Telemetry history includes the current packet; compare against its predecessor."""
        request = PredictionInput(**request_data(speed=30, history=[
            {"event_time": (NOW-timedelta(seconds=10)).isoformat(), "speed": 20},
            {"event_time": NOW.isoformat(), "speed": 30}]))
        self.assertEqual(ml._features(request)[2], 10)

    def test_training_path_is_confined(self):
        """The unauthenticated API cannot read arbitrary host files."""
        for value in ("../README.md", str(ROOT / "README.md")):
            response = self.client.post("/train", json={"labels_path": value})
            self.assertEqual(response.status_code, 400)

    def test_openapi_contract(self):
        """Both inference endpoints publish concrete response schemas."""
        schema = self.client.get("/openapi.json").json()
        self.assertIn("PredictionOutput", schema["components"]["schemas"])
        self.assertIn("BatchPredictionInput", schema["components"]["schemas"])


class IngestionTests(unittest.IsolatedAsyncioTestCase):
    """Exercise backend-to-ML calls across actual ASGI HTTP boundaries."""

    async def asyncSetUp(self):
        """Install an HTTP client talking to the real ML ASGI app in-process."""
        reset_state()
        self.ml_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=ml.app), base_url="http://ml")
        backend.app.state.ml_client = self.ml_client
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url="http://backend")

    async def asyncTearDown(self):
        """Close both connection pools."""
        await self.ml_client.aclose()
        await self.client.aclose()

    async def test_batch_round_trip_no_silent_fallback(self):
        """The backend must send the envelope accepted by the ML service."""
        response = await self.client.post("/api/telemetry/batch", json=[telemetry()])
        self.assertEqual(response.status_code, 200, response.text)
        forecast = response.json()["results"][0]["forecast"]
        self.assertEqual(forecast["model"], "cur_dev_baseline")
        self.assertEqual(forecast["delay_seconds"], 240)
        self.assertEqual(forecast["risk"], "unknown")
        self.assertEqual(backend.metrics["ml_fallbacks"], 0)

    async def test_single_batch_equivalence(self):
        """Single and batch endpoints use the same feature and result paths."""
        single = (await self.client.post("/api/telemetry", json=telemetry())).json()
        reset_state()
        batch = (await self.client.post("/api/telemetry/batch", json=[telemetry()])).json()["results"][0]
        self.assertEqual(single, batch)

    async def test_unknown_vehicle_is_context_only(self):
        """Unknown vehicle packets never call ML or create official stops."""
        response = await self.client.post("/api/telemetry", json=telemetry(tr_id="unknown"))
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["scheduled"])
        self.assertEqual(backend.metrics["predictions"], 0)
        self.assertEqual(set(backend.schedule), {"bus"})

    async def test_batch_isolates_bad_packet(self):
        """One malformed row must not reject all valid rows in the batch."""
        response = await self.client.post("/api/telemetry/batch", json=[None, telemetry()])
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["results"][0]["accepted"])
        self.assertTrue(response.json()["results"][1]["accepted"])

    async def test_nonfinite_packet_returns_validation_error(self):
        """API validation errors containing NaN must still serialize as HTTP 422."""
        response = await self.client.post("/api/telemetry", content=json.dumps(telemetry(lat=float("nan"))),
                                          headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)

    async def test_out_of_order_cannot_overwrite_newer_state(self):
        """Late packets and duplicates must not corrupt history or current position."""
        await self.client.post("/api/telemetry", json=telemetry())
        old = await self.client.post("/api/telemetry", json=telemetry(event_time=(NOW-timedelta(seconds=1)).isoformat()))
        self.assertFalse(old.json()["accepted"])
        self.assertEqual(len(backend.history["bus"]), 1)

    async def test_batch_orders_events_but_preserves_response_order(self):
        """Unsorted batches must not discard a preceding valid observation."""
        response = await self.client.post("/api/telemetry/batch", json=[
            telemetry(event_time=(NOW+timedelta(seconds=1)).isoformat()), telemetry()])
        self.assertTrue(all(row["accepted"] for row in response.json()["results"]))
        self.assertEqual(len(backend.history["bus"]), 2)

    async def test_slow_inference_cannot_overwrite_latest_state(self):
        """An older in-flight forecast may finish after a more recent request."""
        started, release = asyncio.Event(), asyncio.Event()

        async def handler(request):
            """Delay only the older ML response."""
            item = json.loads(request.content)["predictions"][0]
            if utc_time(item["event_time"]) == NOW:
                started.set()
                await release.wait()
            return httpx.Response(200, json={"predictions": [{
                "delay_seconds": item["cur_dev_s"], "delay_probability": .8, "probability_source": "classifier",
                "model": "fixture", "horizon_minutes": (utc_time(item["target_time_begin"])-utc_time(item["event_time"])).total_seconds()/60,
                "target_stop_id": item["target_stop_id"]}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock:
            backend.app.state.ml_client = mock
            earlier = asyncio.create_task(backend.ingest(backend.Telemetry(**telemetry())))
            await started.wait()
            await backend.ingest(backend.Telemetry(**telemetry(
                event_time=(NOW+timedelta(seconds=1)).isoformat(), cur_dev_s=500)))
            release.set()
            await earlier
        self.assertEqual(backend.vehicles["bus"]["forecast"]["delay_seconds"], 500)
        self.assertEqual(len(backend.incidents), 1)

    async def test_malformed_ml_response_degrades_entire_batch(self):
        """Truncated/malformed ML output must not disappear silently through zip()."""
        async def handler(request):
            """Return an invalid empty forecast list."""
            return httpx.Response(200, json={"predictions": []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock:
            backend.app.state.ml_client = mock
            result = await backend.ingest(backend.Telemetry(**telemetry()))
        self.assertEqual(result["forecast"]["model"], "schedule-fallback")
        self.assertEqual(result["forecast"]["risk"], "unknown")

    async def test_ml_disconnect_degrades_without_crash(self):
        """An unreachable ML server preserves known delay without inventing probability."""
        async def handler(request):
            """Simulate a network timeout without external network access."""
            raise httpx.ReadTimeout("timeout", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock:
            backend.app.state.ml_client = mock
            result = await backend.ingest(backend.Telemetry(**telemetry()))
        self.assertTrue(result["accepted"])
        self.assertTrue(result["forecast"]["degraded"])
        self.assertEqual(result["forecast"]["delay_seconds"], 240)

    async def test_replay_switches_schedule_and_blocks_live_ingestion(self):
        """Replay must use its own split schedule and exclude concurrent live packets."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root/"test").mkdir()
            (root/"test/schedule.csv").write_text(
                "tt_action_item_id,time_begin,tr_id,geom,building_address\n"
                "f,2026-01-06 12:12:00,new,POINT (37.6 55.7),A\n", encoding="utf-8")
            (root/"test/traffic.csv").write_text(
                "tr_id,event_time,speed,lat,lon,location_valid\n"
                "new,2026-01-06 12:00:00,20,55.7,37.6,True\n", encoding="utf-8")
            with patch.object(backend, "ROOT", root):
                response = await self.client.post("/api/replay/test")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(set(backend.schedule), {"new"})
            self.assertEqual(response.json()["ingested"], 1)
        backend.replaying = True
        self.assertEqual((await self.client.post("/api/telemetry", json=telemetry())).status_code, 409)
        backend.replaying = False

    async def test_invalid_positions_are_not_drawable(self):
        """GPS-invalid or out-of-range points are discarded, not only flagged."""
        for packet in [telemetry(lat=100), telemetry(location_valid=False)]:
            row = backend._normalize(backend.Telemetry(**packet))
            self.assertIsNone(row["lat"])
            self.assertIsNone(row["lon"])
            self.assertFalse(row["location_valid"])

    async def test_invalid_replay_packet_keeps_last_known_position(self):
        """A bad GPS update must not make the dashboard vehicle disappear."""
        first = backend.Telemetry(**telemetry())
        second = backend.Telemetry(**telemetry(
            event_time=(NOW + timedelta(seconds=30)).isoformat(),
            lat=None, lon=None, location_valid=False,
        ))
        await backend._process_batch([first])
        result = (await backend._process_batch([second]))["results"][0]
        self.assertTrue(result["accepted"])
        vehicle = backend.vehicles["bus"]
        self.assertTrue(vehicle["position_stale"])
        self.assertTrue(vehicle["location_valid"])
        self.assertEqual(vehicle["lat"], 55.7)
        self.assertEqual(vehicle["lon"], 37.6)

    async def test_device_id_is_not_assumed_to_be_vehicle_id(self):
        """Only a known mapping can turn a unit-only packet into a scheduled vehicle."""
        backend.unit_mapping["device"] = "bus"
        result = await backend.ingest(backend.Telemetry(**telemetry(tr_id=None, unit_id="device")))
        self.assertTrue(result["scheduled"])
        reset_state()
        result = await backend.ingest(backend.Telemetry(**telemetry(tr_id=None, unit_id="bus")))
        self.assertFalse(result["scheduled"])

    async def test_vehicle_state_is_bounded(self):
        """Unscheduled vehicle IDs cannot grow memory indefinitely."""
        with patch.object(backend, "MAX_VEHICLES", 2):
            for key in ["one", "two", "three"]:
                await backend.ingest(backend.Telemetry(**telemetry(tr_id=key)))
        self.assertEqual(len(backend.vehicles), 2)
        self.assertNotIn("one", backend.vehicles)

    async def test_dashboard_and_assets_are_served(self):
        """Dashboard HTML and external safe-rendering JS are reachable."""
        self.assertEqual((await self.client.get("/")).status_code, 200)
        script = await self.client.get("/dashboard/app.js")
        self.assertEqual(script.status_code, 200)
        self.assertNotIn(".innerHTML", script.text)

    async def test_ndtp_tcp_server_accepts_real_emulator_frame(self):
        """The backend lifespan binds TCP and processes complete framed Nav00 realtime packets."""
        import struct
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root/"validate").mkdir()
            (root/"validate/schedule_plan.csv").write_text(
                "tt_action_item_id,time_begin,order_date,manual_fill,tr_id,geom,building_address\n"
                "s,2026-01-06 12:12:00,2026-01-06,False,scheduled,POINT (37.6 55.7),Stop\n",
                encoding="utf-8")
            (root/"validate/traffic.csv").write_text(
                "tr_id,unit_id\nscheduled,987\n", encoding="utf-8")
            payload = bytearray(26)
            payload[:4] = int(NOW.timestamp()).to_bytes(4, "little")
            payload[4:8] = int(37.6*10_000_000).to_bytes(4, "little")
            payload[8:12] = int(55.7*10_000_000).to_bytes(4, "little")
            payload[12] = 0b11100000
            cell = bytes([0, 0]) + bytes(payload)
            nph = struct.pack("<HHHI", 1, 101, 1, 1) + cell
            crc = backend.crc16_modbus(nph)
            npl = (struct.pack("<HHH", 0x7E7E, len(nph), 0) + crc.to_bytes(2, "big")
                   + struct.pack("<BIH", 2, 987, 0))
            with patch.object(backend, "ROOT", root), patch.object(backend, "NDTP_PORT", 0):
                async with backend.lifespan(backend.app):
                    port = backend.app.state.ndtp_server.sockets[0].getsockname()[1]
                    reader, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.write(npl+nph)
                    await writer.drain()
                    writer.close()
                    await writer.wait_closed()
                    for _ in range(50):
                        if backend.metrics["packets"]:
                            break
                        await asyncio.sleep(.01)
            self.assertEqual(backend.metrics["packets"], 1)
            self.assertEqual(backend.metrics["unknown_vehicles"], 0)

    async def test_json_irima_doors_and_passenger_counts_are_used(self):
        """Decoded NDTP IRMA cells create door and passenger-derived signals."""
        packet = backend.parse_ndtp_nav({
            "unitId": "device", "tr_id": "bus",
            "G6CellNav00": {"timestamp": int(NOW.timestamp()), "longitude": 376000000,
                            "latitude": 557000000, "flags": 0b11100000},
            "G6CellIrma04": {"irma_present_door1": 1, "irma_closed_door1": 0,
                              "irma_door_in1": 3, "irma_door_out1": 1},
        })
        self.assertTrue(packet.door_open)
        self.assertEqual(packet.door_passenger_count, 4)

    async def test_whatif_does_not_invent_headways(self):
        """What-if scopes the estimate to one official vehicle itinerary."""
        missing = await self.client.post("/api/what-if", json={"tr_id": "unknown"})
        self.assertEqual(missing.status_code, 404)
        result = await self.client.post("/api/what-if", json={"tr_id": "bus", "extra_vehicles": 2})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["projected_vehicles"], 3)
        self.assertIn("Дополнительный выпуск", " ".join(result.json()["assumptions"]))


class DatasetTests(unittest.TestCase):
    """Check actual reference files and point-in-time training joins."""

    def test_unknown_vehicle_counts(self):
        """Validate the mandatory 17 context vehicles in the supplied split."""
        backend.load_schedule(ROOT/"dataset/validate/schedule_plan.csv")
        with (ROOT/"dataset/validate/traffic.csv").open(encoding="utf-8-sig", newline="") as stream:
            identities = {row["tr_id"] for row in csv.DictReader(stream)}
        self.assertEqual(len(identities), 30)
        self.assertEqual(len(identities - set(backend.schedule)), 17)

    def test_validate_target_horizon(self):
        """Every provided forecast point must satisfy the same serving-time contract."""
        with (ROOT/"dataset/validate/points.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                PredictionInput.model_validate({**row, "event_time": row["T"]})


class WarningPatternTests(unittest.TestCase):
    """Cover backend's transparent dispatcher-facing warning pattern rules."""

    def classify(self, **feature_changes):
        """Build a valid prediction request and classify a synthetic evidence snapshot."""
        row_changes = feature_changes.pop("row", {})
        feature_changes.setdefault("cur_dev_s", 0)
        payload = PredictionInput.model_validate(request_data(**feature_changes))
        return backend._classify_warning_pattern(payload, row_changes)

    def test_known_patterns_and_unclassified_fallback(self):
        """Expected sensor signals map to stable codes without calling them confirmed causes."""
        cases = [
            ({"speed_drop_kmh": 18, "row": {"speed": 5}}, "sharp_speed_drop"),
            ({"stopped_seconds": 75, "door_passenger_count": 2}, "extended_boarding_stop"),
            ({"stopped_seconds": 100, "row": {
                "speed": 0, "matched_stop": {"distance_to_stop_m": 30},
            }}, "long_stop_near_scheduled_stop"),
            ({"segment_avg_speed_kmh": 7, "segment_distance_m": 250, "row": {
                "speed": 6, "segment_speed_samples": 5,
            }}, "sustained_low_segment_speed"),
            ({"cur_dev_s": 180}, "observed_schedule_lateness"),
            ({"nearby_context": 4, "cur_dev_s": 0}, "nearby_vehicle_density"),
            ({"heading_error_deg": 80, "row": {"speed": 12}}, "heading_route_mismatch"),
            ({}, "no_single_pattern"),
        ]
        for features, expected_code in cases:
            with self.subTest(pattern=expected_code):
                self.assertEqual(self.classify(**features)["code"], expected_code)


    def test_training_and_submission_use_same_schedule_segment_matcher(self):
        """Submission uses the identical ML feature builder as training, not Backend HMM state."""
        from scripts.create_submission import generate
        from ml_service.app.main import _build_schedule_segment_index, _map_match_features
        from unittest.mock import Mock
        stops = [
            {"id": "a", "time": NOW, "lat": 55.70, "lon": 37.60},
            {"id": "b", "time": NOW+timedelta(minutes=10), "lat": 55.71, "lon": 37.60},
            {"id": "c", "time": NOW+timedelta(minutes=20), "lat": 55.72, "lon": 37.60},
        ]
        for index, item in enumerate(stops):
            item["index"] = index
        validate_time = NOW + timedelta(minutes=5)
        current = {"lat": 55.705, "lon": 37.6001, "heading": 0}
        indexed = _build_schedule_segment_index({"bus": stops})["bus"]
        built = _map_match_features(
            stops, current["lat"], current["lon"], validate_time, 0,
            spatial_index=indexed,
        )
        exhaustive = _map_match_features(
            stops, current["lat"], current["lon"], validate_time, 0,
        )
        self.assertEqual(built, exhaustive)
        self.assertAlmostEqual(built["segment_progress"], .5, delta=.03)
        self.assertGreater(built["segment_distance_m"], 0)
        later_time = NOW + timedelta(minutes=15)
        later_indexed = _map_match_features(
            stops, 55.715, 37.6001, later_time, 0, spatial_index=indexed,
        )
        later_exhaustive = _map_match_features(
            stops, 55.715, 37.6001, later_time, 0,
        )
        self.assertEqual(later_indexed, later_exhaustive)
        self.assertEqual(later_indexed["segment_index"], 1)

    def test_temporal_stop_disambiguation_and_distance_gate(self):
        """Repeated geometry must select the relevant visit; remote points remain unmatched."""
        reset_state()
        match = backend.match_stop("bus", 55.7, 37.6, NOW, 240)
        self.assertEqual(match["stop_id"], "past")
        self.assertIsNone(backend.match_stop("bus", 0, 0, NOW))

    def test_schedule_segment_matching_uses_heading_and_progress(self):
        """Map matching chooses an official segment and estimates its projection."""
        reset_state()
        left, right = stop("a", NOW), stop("b", NOW + timedelta(minutes=2))
        right.update({"_index": 1, "_lat": 55.71, "_lon": 37.6})
        backend.schedule["bus"] = [left, right]
        match = backend.match_stop("bus", 55.705, 37.6001, NOW + timedelta(minutes=1),
                                   heading=0, speed=25)
        self.assertEqual(match["segment_index"], 0)
        self.assertAlmostEqual(match["segment_progress"], .5, delta=.03)
        self.assertLess(match["heading_error_deg"], 2)
        self.assertIsNone(backend.match_stop("bus", 55.705, 37.61, NOW))

    def test_map_matching_handles_previous_unmatched_sequence_progress(self):
        """A prior packet with an explicit None progress value must not crash matching."""
        reset_state()
        left = stop("past", NOW - timedelta(minutes=4))
        right = stop("future", NOW + timedelta(minutes=12))
        right.update({"_index": 1, "_lat": 55.71, "_lon": 37.6})
        backend.schedule["bus"] = [left, right]
        backend.segment_index["bus"] = {}
        prior = {
            "lat": 55.704, "lon": 37.6, "event_time": NOW.isoformat(),
            "matched_stop_id": None, "sequence_progress": None,
        }
        result = backend.match_stop(
            "bus", 55.705, 37.6, NOW + timedelta(seconds=30),
            heading=0, speed=10, history_points=[prior],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["segment_index"], 0)

    def test_ndtp_g6cell_nav_decodes_sign_validity_and_speed(self):
        """Decode the documented 26-byte Nav00 cell with hemisphere/validity flags."""
        from backend.app.main import decode_ndtp_nav_cell
        payload = bytearray(26)
        payload[0:4] = int(NOW.timestamp()).to_bytes(4, "little")
        payload[4:8] = (376173210).to_bytes(4, "little")
        payload[8:12] = (557551234).to_bytes(4, "little")
        payload[12] = (1 << 5) | (1 << 6) | (1 << 7)
        payload[14:16] = (32).to_bytes(2, "little")
        payload[18:20] = (90).to_bytes(2, "little")
        packet = decode_ndtp_nav_cell(bytes(payload), unit_id="device-1", tr_id="bus")
        self.assertAlmostEqual(packet.lat, 55.7551234, places=6)
        self.assertAlmostEqual(packet.lon, 37.617321, places=6)
        self.assertEqual(packet.speed, 32)
        self.assertEqual(packet.heading, 90)
        self.assertTrue(packet.location_valid)
        payload[12] = 0
        reversed_fix = decode_ndtp_nav_cell(bytes(payload), unit_id="device-1")
        self.assertLess(reversed_fix.lat, 0)
        self.assertLess(reversed_fix.lon, 0)
        with self.assertRaises(ValueError):
            decode_ndtp_nav_cell(b"short", "device-1")

    def test_ndtp_frame_decoder_accepts_crc_and_realtime_nav(self):
        """NPL/NPH framing and CRC are validated before Nav00 enters the stream."""
        import struct
        from backend.app.main import crc16_modbus, decode_ndtp_frame
        body = bytearray(28)
        body[0] = 0
        body[1] = 0
        nav = bytearray(26)
        nav[0:4] = int(NOW.timestamp()).to_bytes(4, "little")
        nav[4:8] = (376173210).to_bytes(4, "little")
        nav[8:12] = (557551234).to_bytes(4, "little")
        nav[12] = (1 << 5) | (1 << 6) | (1 << 7)
        body[2:] = nav
        nph = struct.pack("<HHHI", 1, 101, 1, 7) + bytes(body)
        crc = crc16_modbus(nph)
        npl = struct.pack("<HHH", 0x7E7E, len(nph), 0) + crc.to_bytes(2, "big") + struct.pack("<BIH", 2, 123, 0)
        kind, packet = decode_ndtp_frame(npl + nph)
        self.assertEqual(kind, "realtime")
        self.assertEqual(packet.unit_id, "123")
        self.assertTrue(packet.location_valid)
        corrupted = bytearray(npl + nph)
        corrupted[-1] ^= 1
        with self.assertRaises(ValueError):
            decode_ndtp_frame(bytes(corrupted))

    def test_json_door_cell_is_normalized(self):
        """Decoded IRMA/Crown JSON cells produce door and passenger features."""
        packet = backend.parse_ndtp_nav({
            "unitId": "1", "tr_id": "bus",
            "G6CellNav00": {
                "timestamp": int(NOW.timestamp()), "longitude": 376173210,
                "latitude": 557551234, "flags": 0b11100000, "speedAvg": 2, "course": 0,
            },
            "G6CellIrma04": {
                "irma_present_door1": 1, "irma_closed_door1": 0,
                "irma_door_in1": 3, "irma_door_out1": 1,
            },
        })
        self.assertTrue(packet.door_open)
        self.assertEqual(packet.door_passenger_count, 4)

    def test_join_includes_equal_timestamp_and_excludes_future(self):
        """String comparison previously missed an event at T with fractional zero seconds."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            (root/"train").mkdir()
            (root/"labels").mkdir()
            (root/"train/schedule.csv").write_text(
                "tr_id,tt_action_item_id,time_begin,geom\n"
                "bus,future,2026-01-06 12:12:00,POINT (37.6 55.7)\n", encoding="utf-8")
            (root/"train/traffic.csv").write_text(
                "tr_id,event_time,speed,heading,lat,lon,location_valid\n"
                "bus,2026-01-06 12:00:00.000000,20,90,55.7,37.6,True\n"
                "bus,2026-01-06 12:00:01,99,90,55.7,37.6,True\n", encoding="utf-8")
            (root/"labels/labels_train.csv").write_text(
                "tr_id,T,target_time_begin,target_stop_id,cur_dev_s,target_delay_s\n"
                "bus,2026-01-06 12:00:00,2026-01-06 12:12:00,future,240,250\n", encoding="utf-8")
            with patch.object(ml, "DATASET_DIR", root):
                rows = ml.dataset_rows()
            self.assertEqual(rows[0]["speed"], 20)
            self.assertEqual(len(rows[0]["history"]), 1)

    def test_replay_sorting(self):
        """CSV row order must not cause future observations to enter earlier forecasts."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp)/"traffic.csv"
            path.write_text("tr_id,event_time,speed,lat,lon,location_valid\n"
                            "bus,2026-01-06 12:00:01,20,55.7,37.6,True\n"
                            "bus,2026-01-06 12:00:00,10,55.7,37.6,True\n", encoding="utf-8")
            packets, rejected = backend._read_replay(path)
            self.assertEqual(rejected, 0)
            self.assertLess(packets[0].event_time, packets[1].event_time)


class TrainingTests(unittest.TestCase):
    """Ensure reported metrics belong to the served ensemble and failed publication is safe."""

    def setUp(self):
        """Create deterministic chronological samples with both delay classes."""
        reset_state()
        self.rows = []
        for i in range(60):
            moment = NOW + timedelta(minutes=i * 5)
            self.rows.append({
                **request_data(event_time=moment.isoformat(),
                               target_time_begin=(moment+timedelta(minutes=12)).isoformat(),
                               cur_dev_s=100 + i),
                "target_delay_s": float(50 + i * 3),
            })

    def test_metrics_reload_and_failed_publication(self):
        """Holdout metric is finite; failed publication leaves the old model serving."""
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            folder = Path(tmp)
            # Exercise the baseline branch deterministically without requiring optional downloads.
            with patch.object(ml, "MODEL_DIR", folder), patch.object(ml, "MODEL_PATH", folder/"model.joblib"), \
                    patch.dict("sys.modules", {"catboost": None, "torch": None}):
                report = ml._fit(self.rows)
                self.assertTrue(np.isfinite(report["validation_mae_seconds"]))
                self.assertGreater(report["purged_samples"], 0)
                before = ml.predict(PredictionInput.model_validate(self.rows[-1])).model_dump()
                ml.startup()
                self.assertEqual(before, ml.predict(PredictionInput.model_validate(self.rows[-1])).model_dump())
                previous = ml.model
                with patch("joblib.dump", side_effect=OSError("disk full")):
                    with self.assertRaises(OSError):
                        ml._fit(self.rows)
                self.assertIs(ml.model, previous)
                self.assertFalse(ml.training_lock.locked())
                self.assertEqual(list(folder.glob(".*.joblib")), [])

    def test_selected_regressor_is_served_without_catboost_or_cnn_blend(self):
        """Delay inference is deterministic and uses the MAE-selected LightGBM branch."""
        from sklearn.dummy import DummyRegressor
        x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        bundle = {
            "regressor": DummyRegressor(strategy="constant", constant=111).fit(x, [0, 0]),
            "lgbm_regressor": DummyRegressor(strategy="constant", constant=22).fit(x, [0, 0]),
            "cat_regressor": DummyRegressor(strategy="constant", constant=999).fit(x, [0, 0]),
            "classifier": None, "cat_classifier": None, "neural": None,
            "backend": "lightgbm+catboost+pytorch", "features": ml.FEATURES,
        }
        with patch.object(ml, "onnx_session", None), patch.object(ml, "onnx_input_name", None):
            prediction, _ = ml._ensemble(bundle, x)
        self.assertTrue(np.array_equal(prediction, np.asarray([22.0, 22.0])))


if __name__ == "__main__":
    unittest.main()
