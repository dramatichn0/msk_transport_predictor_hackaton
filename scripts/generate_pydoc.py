"""Generate standalone HTML documentation for the project's Python modules."""
from __future__ import annotations

import os
import pydoc
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "pydoc"
MODULES = (
    "backend.app.main",
    "ml_service.app.main",
    "transit_common.contracts",
)


def main() -> int:
    """Generate PyDoc HTML pages under docs/pydoc."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    previous = Path.cwd()
    try:
        os.chdir(OUTPUT)
        for module in MODULES:
            pydoc.writedoc(module)
            filename = OUTPUT / f"{module}.html"
            if not filename.exists():
                raise RuntimeError(f"Could not generate documentation for {module}")
            html = filename.read_text(encoding="utf-8")
            html = re.sub(
                r'(<td class="extra">)<a href="\."[^>]*>index</a><br>'
                r'<a href="file:[^"]*">[^<]*</a>',
                r'\1<a href=".">index</a>',
                html,
            )
            filename.write_text(html, encoding="utf-8")
            print(f"{module}: {filename}")
    finally:
        os.chdir(previous)
    index = OUTPUT / "README.md"
    index.write_text(
        "# PyDoc проекта\n\n"
        "Сгенерированные страницы:\n\n"
        + "\n".join(f"- [{module}]({module}.html)" for module in MODULES)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
