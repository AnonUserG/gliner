import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests requiring the real model "
        "(skip with -m 'not integration')",
    )


# ---------------------------------------------------------------------------
# Canned mock responses
# ---------------------------------------------------------------------------

_MOCK_BATCH = [
    [
        {"text": "Angela Merkel", "label": "person", "start": 0, "end": 13, "score": 0.98},
        {"text": "Berlin", "label": "city", "start": 22, "end": 28, "score": 0.95},
    ],
    [
        {"text": "Paris", "label": "city", "start": 6, "end": 11, "score": 0.96},
    ],
]


# ---------------------------------------------------------------------------
# Unit-test fixtures  (function-scoped — fast, no real model)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.predict_entities.return_value = _MOCK_BATCH[0]
    mock.batch_predict_entities.return_value = _MOCK_BATCH
    return mock


@pytest.fixture
def client(monkeypatch, mock_model):
    """TestClient backed by a mock model — no torch/gliner download."""
    import app as app_module

    # Patch the loader so the lifespan receives our mock instead of the real model
    monkeypatch.setattr(app_module, "_load_model", lambda: mock_model)

    with TestClient(app_module.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Integration-test fixture  (module-scoped — loads the real model once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def integration_client():
    """TestClient backed by the real GLiNER model.

    Requires model weights to be present (inside Docker image or downloaded
    locally).  Run only with:  pytest -m integration
    """
    from app import app

    with TestClient(app) as c:
        yield c
