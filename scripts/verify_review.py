"""Reproduce historical verification without modifying active model artifacts."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from backend.app import main as backend
from ml_service.app import main as ml

ROOT = Path(__file__).resolve().parents[1]


async def replay_validate() -> dict:
    """Replay the actual validation split through backend-to-ML ASGI HTTP calls."""
    started = time.perf_counter()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=ml.app), base_url="http://ml"
    ) as client:
        backend.app.state.ml_client = client
        result = await backend.replay("validate")
    return {**result, "seconds": round(time.perf_counter() - started, 3),
            "metrics": dict(backend.metrics),
            "state_vehicles": len(backend.vehicles),
            "scheduled_state_vehicles": sum(v["scheduled"] for v in backend.vehicles.values())}


def main() -> None:
    """Run real-data regression checks; write an auditable JSON report."""
    rows = ml.dataset_rows()
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        folder = Path(temp)
        with patch.object(ml, "MODEL_DIR", folder), patch.object(ml, "MODEL_PATH", folder / "model.joblib"):
            scores = ml._fit(rows)
            before = ml.predict(ml.PredictionInput.model_validate(rows[-1])).model_dump()
            ml.startup()
            after = ml.predict(ml.PredictionInput.model_validate(rows[-1])).model_dump()
            reloaded = before == after
            # Replay tests protocol/chronology rather than claiming accuracy without validate labels.
            ml.model = None
            replay = asyncio.run(replay_validate())
    report = {
        "dependencies": {name: bool(importlib.util.find_spec(name)) for name in ("catboost", "torch")},
        "holdout": scores, "reload_equal": reloaded, "validate_replay_baseline": replay,
        "notes": ["In-process ASGI HTTP transport, not Docker or network latency benchmark.",
                  "Holdout metric evaluates the exact ensemble; validate has no ground truth.",
                  "Temp artifacts removed; active models are not changed."],
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/review-verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
