"""
Unit tests — no real model loaded.
Run with:  pytest -m "not integration"
"""


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_shape(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["model"], str) and data["model"]
    assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# /extract — validation
# ---------------------------------------------------------------------------

class TestExtractValidation:
    def test_missing_labels_field(self, client):
        resp = client.post("/extract", json={"text": "hello"})
        assert resp.status_code == 422

    def test_empty_labels_list(self, client):
        resp = client.post("/extract", json={"text": "hello", "labels": []})
        assert resp.status_code == 422

    def test_threshold_above_one(self, client):
        resp = client.post(
            "/extract", json={"text": "hello", "labels": ["person"], "threshold": 1.1}
        )
        assert resp.status_code == 422

    def test_threshold_below_zero(self, client):
        resp = client.post(
            "/extract", json={"text": "hello", "labels": ["person"], "threshold": -0.1}
        )
        assert resp.status_code == 422

    def test_missing_text_field(self, client):
        resp = client.post("/extract", json={"labels": ["person"]})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /extract — behaviour
# ---------------------------------------------------------------------------

def test_empty_text_returns_empty_entities(client):
    resp = client.post("/extract", json={"text": "", "labels": ["person"]})
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


def test_extract_response_shape(client):
    resp = client.post(
        "/extract",
        json={"text": "Angela Merkel visited Berlin.", "labels": ["person", "city"]},
    )
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert len(entities) > 0
    for e in entities:
        assert isinstance(e["text"], str)
        assert isinstance(e["label"], str)
        assert isinstance(e["start"], int)
        assert isinstance(e["end"], int)
        assert isinstance(e["score"], float)
        assert e["start"] >= 0
        assert e["end"] > e["start"]


def test_extract_threshold_boundary_values(client):
    for threshold in (0.0, 1.0):
        resp = client.post(
            "/extract",
            json={"text": "test", "labels": ["person"], "threshold": threshold},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /extract_batch — validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    def test_missing_labels(self, client):
        resp = client.post("/extract_batch", json={"texts": ["hello"]})
        assert resp.status_code == 422

    def test_empty_labels(self, client):
        resp = client.post(
            "/extract_batch", json={"texts": ["hello"], "labels": []}
        )
        assert resp.status_code == 422

    def test_missing_texts(self, client):
        resp = client.post("/extract_batch", json={"labels": ["person"]})
        assert resp.status_code == 422

    def test_empty_texts_list_returns_empty_results(self, client):
        resp = client.post(
            "/extract_batch", json={"texts": [], "labels": ["person"]}
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []


# ---------------------------------------------------------------------------
# /extract_batch — behaviour
# ---------------------------------------------------------------------------

def test_batch_order_preserved(client, mock_model):
    mock_model.batch_predict_entities.return_value = [
        [{"text": "Angela Merkel", "label": "person", "start": 0, "end": 13, "score": 0.98}],
        [{"text": "Paris", "label": "city", "start": 6, "end": 11, "score": 0.96}],
    ]
    texts = ["Angela Merkel is here.", "Visit Paris today."]
    resp = client.post(
        "/extract_batch", json={"texts": texts, "labels": ["person", "city"]}
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    # First text → person, second text → city
    assert results[0]["entities"][0]["label"] == "person"
    assert results[1]["entities"][0]["label"] == "city"


def test_batch_response_length_matches_input(client, mock_model):
    mock_model.batch_predict_entities.return_value = [[], [], []]
    resp = client.post(
        "/extract_batch",
        json={"texts": ["a", "b", "c"], "labels": ["person"]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 3
