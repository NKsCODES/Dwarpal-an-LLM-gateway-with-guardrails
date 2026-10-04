"""Start the gateway API and the dashboard together.

    python run.py

API on http://127.0.0.1:8000 (docs at /docs), dashboard on http://127.0.0.1:8501.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    api_port = os.getenv("GATEWAY_PORT", "8000")
    ui_port = os.getenv("DASHBOARD_PORT", "8501")
    env = {**os.environ, "GATEWAY_API_URL": f"http://127.0.0.1:{api_port}"}
    commands = [
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", api_port],
        [sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.port", ui_port, "--server.headless", "true"],
    ]
    procs = [subprocess.Popen(cmd, cwd=ROOT, env=env) for cmd in commands]
    print(f"\nAPI:       http://127.0.0.1:{api_port}/docs\nDashboard: http://127.0.0.1:{ui_port}\nPress Ctrl+C to stop.\n")

    def stop(*_: object) -> None:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while all(proc.poll() is None for proc in procs):
            time.sleep(0.5)
    finally:
        stop()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return max((proc.returncode or 0) for proc in procs)


if __name__ == "__main__":
    raise SystemExit(main())
