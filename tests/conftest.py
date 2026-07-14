"""Shared pytest fixtures: model artifacts and a FastAPI TestClient, both
loaded once per test session against the real saved models/ artifacts --
these are integration tests against the actual trained models, not mocks,
since the whole point is to verify the persisted artifacts still work.
"""

import pytest
from fastapi.testclient import TestClient

from app import model_loader
from app.api import app


@pytest.fixture(scope="session")
def artifacts():
    return model_loader.get_artifacts()


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
