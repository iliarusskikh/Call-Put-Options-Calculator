"""CLI entrypoint: launch the Streamlit Options Desk app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the Streamlit UI via ``streamlit run``."""
    app_path = Path(__file__).resolve().parent / "options_app" / "app.py"
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.headless",
                "true",
            ]
        )
    )


if __name__ == "__main__":
    main()
