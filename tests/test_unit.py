"""
Unit tests — no real model loaded.
Run with:  pytest -m "not integration"
"""
import logging

from app import _chunk_text


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

    def test_long_text_accepted_when_no_limit_configured(self, client):
        resp = client.post(
            "/extract", json={"text": "x" * 50_000, "labels": ["person"]}
        )
        assert resp.status_code == 200


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

    def test_long_text_in_batch_accepted_when_no_limit_configured(self, client, mock_model):
        mock_model.batch_predict_entities.return_value = [[], []]
        resp = client.post(
            "/extract_batch",
            json={"texts": ["ok text", "x" * 50_000], "labels": ["person"]},
        )
        assert resp.status_code == 200

    def test_large_batch_accepted_when_no_limit_configured(self, client, mock_model):
        texts = ["text"] * 1000
        mock_model.batch_predict_entities.return_value = [[] for _ in texts]
        resp = client.post(
            "/extract_batch", json={"texts": texts, "labels": ["person"]}
        )
        assert resp.status_code == 200


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


# ---------------------------------------------------------------------------
# _chunk_text — pure helper
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_empty_text(self):
        assert _chunk_text("", chunk_size=300) == [(0, 0)]

    def test_short_text_is_single_chunk(self):
        text = "Hello world, this is a test."
        chunks = _chunk_text(text, chunk_size=300)
        assert chunks == [(0, len(text))]

    def test_splits_into_multiple_chunks(self):
        words = [f"word{i}" for i in range(10)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=3)
        # 10 words / 3 per chunk -> 4 chunks (3, 3, 3, 1)
        assert len(chunks) == 4

    def test_exact_multiple_of_chunk_size(self):
        text = " ".join(f"word{i}" for i in range(6))
        chunks = _chunk_text(text, chunk_size=3)
        assert len(chunks) == 2

    def test_chunks_reconstruct_original_words_in_order(self):
        words = [f"w{i}" for i in range(7)]
        text = " ".join(words)
        chunks = _chunk_text(text, chunk_size=3)
        assert len(chunks) == 3  # 3, 3, 1
        extracted = []
        for start, end in chunks:
            extracted.extend(text[start:end].split())
        assert extracted == words

    def test_chunks_are_contiguous_and_in_bounds(self):
        text = " ".join(f"word{i}" for i in range(10))
        chunks = _chunk_text(text, chunk_size=3)
        for start, end in chunks:
            assert 0 <= start < end <= len(text)


# ---------------------------------------------------------------------------
# Long-text chunking — /extract and /extract_batch
# ---------------------------------------------------------------------------

class TestChunking:
    def test_extract_chunks_long_text_and_merges_offsets(self, client, mock_model, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "CHUNK_SIZE_WORDS", 2)

        text = "Angela Merkel visited Berlin today"
        # chunk1 "Angela Merkel" (0-13), chunk2 "visited Berlin" (14-28), chunk3 "today" (29-34)
        mock_model.batch_predict_entities.return_value = [
            [{"text": "Angela Merkel", "label": "person", "start": 0, "end": 13, "score": 0.98}],
            [{"text": "Berlin", "label": "city", "start": 8, "end": 14, "score": 0.95}],
            [],
        ]

        resp = client.post("/extract", json={"text": text, "labels": ["person", "city"]})
        assert resp.status_code == 200

        sent_texts = mock_model.batch_predict_entities.call_args[0][0]
        assert sent_texts == ["Angela Merkel", "visited Berlin", "today"]

        entities = resp.json()["entities"]
        person = next(e for e in entities if e["label"] == "person")
        city = next(e for e in entities if e["label"] == "city")

        assert (person["start"], person["end"]) == (0, 13)
        assert text[person["start"]:person["end"]] == "Angela Merkel"

        assert (city["start"], city["end"]) == (22, 28)
        assert text[city["start"]:city["end"]] == "Berlin"

    def test_extract_batch_chunks_one_text_in_single_model_call(self, client, mock_model, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "CHUNK_SIZE_WORDS", 2)

        texts = ["short text", "Angela Merkel visited Berlin today"]
        # text 0: "short text" -> 1 chunk
        # text 1: 5 words -> 3 chunks: "Angela Merkel", "visited Berlin", "today"
        mock_model.batch_predict_entities.return_value = [
            [],
            [{"text": "Angela Merkel", "label": "person", "start": 0, "end": 13, "score": 0.98}],
            [{"text": "Berlin", "label": "city", "start": 8, "end": 14, "score": 0.95}],
            [],
        ]

        resp = client.post(
            "/extract_batch", json={"texts": texts, "labels": ["person", "city"]}
        )
        assert resp.status_code == 200

        # all chunks across both texts go through a single batched call
        assert mock_model.batch_predict_entities.call_count == 1
        sent_texts = mock_model.batch_predict_entities.call_args[0][0]
        assert sent_texts == ["short text", "Angela Merkel", "visited Berlin", "today"]

        results = resp.json()["results"]
        assert results[0]["entities"] == []

        entities1 = results[1]["entities"]
        assert len(entities1) == 2
        person = next(e for e in entities1 if e["label"] == "person")
        city = next(e for e in entities1 if e["label"] == "city")
        assert (person["start"], person["end"]) == (0, 13)
        assert (city["start"], city["end"]) == (22, 28)

    def test_chunking_logs_split_and_merge(self, client, mock_model, monkeypatch, caplog):
        import app as app_module
        monkeypatch.setattr(app_module, "CHUNK_SIZE_WORDS", 2)

        text = "Angela Merkel visited Berlin today"
        mock_model.batch_predict_entities.return_value = [[], [], []]

        with caplog.at_level(logging.INFO):
            resp = client.post("/extract", json={"text": text, "labels": ["person"]})

        assert resp.status_code == 200
        assert "split into 3 chunks" in caplog.text
        assert "chunk 1/3" in caplog.text
        assert "chunk 2/3" in caplog.text
        assert "chunk 3/3" in caplog.text
        assert "merged 0 entities from 3 chunks" in caplog.text

    def test_short_text_does_not_log_chunking(self, client, mock_model, caplog):
        with caplog.at_level(logging.INFO):
            resp = client.post("/extract", json={"text": "short text", "labels": ["person"]})

        assert resp.status_code == 200
        assert "split into" not in caplog.text
        assert "chunk" not in caplog.text
