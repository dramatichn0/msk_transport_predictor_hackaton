"""Validate a generated submission against the supplied sample template."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Check schema, delimiter, full ordered ID coverage, uniqueness, and finite values."""
    candidate = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "submission.csv"
    template_path = ROOT / "dataset" / "sample_submission.csv"
    with candidate.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        if reader.fieldnames != ["sample_id", "prediction"]:
            raise SystemExit(f"Неверные колонки: {reader.fieldnames!r}")
        rows = list(reader)
    with template_path.open(encoding="utf-8-sig", newline="") as stream:
        expected = [row["sample_id"] for row in csv.DictReader(stream, delimiter=";")]
    actual = [row["sample_id"] for row in rows]
    if len(actual) != len(set(actual)):
        raise SystemExit("В CSV есть повторяющиеся sample_id")
    if actual != expected:
        raise SystemExit(f"sample_id не совпадают с шаблоном: строк {len(actual)}, ожидалось {len(expected)}")
    try:
        values = [float(row["prediction"]) for row in rows]
    except (TypeError, ValueError):
        raise SystemExit("prediction должен быть числом в каждой строке") from None
    if not all(math.isfinite(value) for value in values):
        raise SystemExit("prediction содержит NaN или бесконечность")
    print(f"OK: {len(rows)} прогнозов, IDs совпадают, значения конечные. Файл: {candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
