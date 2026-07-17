import os

os.environ["PDM_ENV"] = "testing"

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
