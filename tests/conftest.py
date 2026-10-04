from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from mesh.config import Settings  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    # No simulated latency so the suite runs in a second or two.
    return dataclasses.replace(Settings(), mock_llm=True, mock_latency_scale=0.0, api_key=None, admin_key=None)


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
