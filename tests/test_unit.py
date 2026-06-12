"""
Unit tests — no real model loaded.
Run with:  pytest -m "not integration"
"""
import logging
from unittest.mock import MagicMock

import requests

from app import _chunk_text, _clean_labels, _clean_text, _QuietZipkinExporter


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
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint_exposes_prometheus_format(client):
    client.get("/health")  # generate at least one request to instrument

    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "http_requests_total" in resp.text


# ---------------------------------------------------------------------------
# Trace ID logging
# ---------------------------------------------------------------------------

def test_logs_include_trace_id(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post(
            "/extract",
            json={"text": "Angela Merkel visited Berlin.", "labels": ["person", "city"]},
        )

    records = [r for r in caplog.records if "extract: received" in r.getMessage()]
    assert records
    trace_id = records[0].trace_id
    assert trace_id != "-"
    assert int(trace_id, 16)  # 32-char hex string
    assert len(trace_id) == 32


def test_trace_id_differs_across_requests(client, caplog):
    with caplog.at_level(logging.INFO):
        client.post("/extract", json={"text": "a", "labels": ["person"]})
        client.post("/extract", json={"text": "b", "labels": ["person"]})

    records = [r for r in caplog.records if "extract: received" in r.getMessage()]
    assert len(records) == 2
    assert records[0].trace_id != records[1].trace_id


def test_log_without_request_has_placeholder_trace_id(caplog):
    with caplog.at_level(logging.INFO, logger="app"):
        logging.getLogger("app").info("outside any request")

    records = [r for r in caplog.records if r.getMessage() == "outside any request"]
    assert records[0].trace_id == "-"


# ---------------------------------------------------------------------------
# _QuietZipkinExporter
# ---------------------------------------------------------------------------

class TestQuietZipkinExporter:
    def test_logs_unreachable_once(self, caplog):
        exporter = _QuietZipkinExporter(endpoint="http://zipkin.invalid:9411/api/v2/spans")
        exporter.session.post = MagicMock(
            side_effect=requests.exceptions.ConnectionError("boom")
        )

        with caplog.at_level(logging.WARNING):
            exporter.export([])
            exporter.export([])

        warnings = [r for r in caplog.records if "unreachable" in r.getMessage()]
        assert len(warnings) == 1

    def test_logs_recovery_once(self, caplog):
        exporter = _QuietZipkinExporter(endpoint="http://zipkin:9411/api/v2/spans")
        exporter._unreachable = True
        exporter.session.post = MagicMock(return_value=MagicMock(status_code=202))

        with caplog.at_level(logging.INFO):
            result = exporter.export([])

        from opentelemetry.sdk.trace.export import SpanExportResult
        assert result == SpanExportResult.SUCCESS
        assert exporter._unreachable is False
        recovered = [r for r in caplog.records if "reachable again" in r.getMessage()]
        assert len(recovered) == 1


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

    def test_large_labels_list_accepted_when_no_limit_configured(self, client, mock_model):
        labels = [f"label{i}" for i in range(1000)]
        mock_model.batch_predict_entities.return_value = [[]]
        resp = client.post(
            "/extract", json={"text": "hello", "labels": labels}
        )
        assert resp.status_code == 200

    def test_whitespace_only_labels_rejected(self, client):
        resp = client.post(
            "/extract", json={"text": "hello", "labels": ["  ", "\t"]}
        )
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
# _clean_text — pure helper
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_html_tags(self):
        assert _clean_text("<b>Hello</b> world") == "Hello world"

    def test_strips_tags_without_joining_adjacent_words(self):
        assert _clean_text("Hello<br/>World") == "Hello World"

    def test_decodes_html_entities(self):
        assert _clean_text("Tom &amp; Jerry") == "Tom & Jerry"

    def test_decodes_entities_that_spell_out_tags(self):
        assert _clean_text("&lt;b&gt;Hello&lt;/b&gt;") == "Hello"

    def test_collapses_whitespace_and_strips(self):
        assert _clean_text("  Hello   world  \n\n") == "Hello world"

    def test_nbsp_becomes_regular_space(self):
        assert _clean_text("Hello" + chr(0xA0) + "World") == "Hello World"

    def test_removes_zero_width_chars(self):
        text = "Hello" + chr(0x200B) + chr(0x200C) + chr(0x200D) + "World"
        assert _clean_text(text) == "HelloWorld"

    def test_removes_control_chars_bom_and_soft_hyphen(self):
        text = chr(0xFEFF) + "Hello" + chr(0xAD) + "World" + chr(0x00)
        assert _clean_text(text) == "HelloWorld"

    def test_nfkc_normalizes_fullwidth_chars(self):
        fullwidth_abc = chr(0xFF21) + chr(0xFF22) + chr(0xFF23)
        assert _clean_text(fullwidth_abc) == "ABC"

    def test_empty_input(self):
        assert _clean_text("") == ""

    def test_whitespace_only_input_becomes_empty(self):
        assert _clean_text("   \n\t  ") == ""


# ---------------------------------------------------------------------------
# Text cleaning — /extract and /extract_batch
# ---------------------------------------------------------------------------

class TestCleanTextIntegration:
    def test_extract_cleans_text_before_inference(self, client, mock_model):
        mock_model.batch_predict_entities.return_value = [[]]
        resp = client.post(
            "/extract",
            json={"text": "<p>Angela&nbsp;Merkel</p>", "labels": ["person"]},
        )
        assert resp.status_code == 200
        sent_texts = mock_model.batch_predict_entities.call_args[0][0]
        assert sent_texts == ["Angela Merkel"]

    def test_extract_returns_empty_for_text_that_cleans_to_empty(self, client):
        resp = client.post("/extract", json={"text": "<br/>", "labels": ["person"]})
        assert resp.status_code == 200
        assert resp.json()["entities"] == []

    def test_extract_batch_cleans_each_text(self, client, mock_model):
        mock_model.batch_predict_entities.return_value = [[], []]
        resp = client.post(
            "/extract_batch",
            json={"texts": ["<b>Hello</b>", "World &amp; Co"], "labels": ["person"]},
        )
        assert resp.status_code == 200
        sent_texts = mock_model.batch_predict_entities.call_args[0][0]
        assert sent_texts == ["Hello", "World & Co"]


# ---------------------------------------------------------------------------
# _clean_labels — pure helper
# ---------------------------------------------------------------------------

class TestCleanLabels:
    def test_strips_whitespace(self):
        assert _clean_labels([" person ", "city"]) == ["person", "city"]

    def test_deduplicates_preserving_order(self):
        assert _clean_labels(["person", "city", "person"]) == ["person", "city"]

    def test_deduplicates_after_stripping(self):
        assert _clean_labels(["person", " person "]) == ["person"]

    def test_drops_empty_and_whitespace_only_entries(self):
        assert _clean_labels(["person", "  ", "", "city"]) == ["person", "city"]

    def test_all_empty_returns_empty_list(self):
        assert _clean_labels(["  ", ""]) == []

    def test_is_case_sensitive(self):
        assert _clean_labels(["city", "City"]) == ["city", "City"]


# ---------------------------------------------------------------------------
# Labels cleaning — /extract and /extract_batch
# ---------------------------------------------------------------------------

class TestCleanLabelsIntegration:
    def test_duplicate_and_whitespace_labels_collapse(self, client, mock_model):
        mock_model.batch_predict_entities.return_value = [[]]
        resp = client.post(
            "/extract",
            json={"text": "Angela Merkel", "labels": ["person", " person ", "person"]},
        )
        assert resp.status_code == 200
        sent_labels = mock_model.batch_predict_entities.call_args[0][1]
        assert sent_labels == ["person"]

    def test_batch_duplicate_labels_collapse(self, client, mock_model):
        mock_model.batch_predict_entities.return_value = [[], []]
        resp = client.post(
            "/extract_batch",
            json={"texts": ["a", "b"], "labels": ["city", "person", "city"]},
        )
        assert resp.status_code == 200
        sent_labels = mock_model.batch_predict_entities.call_args[0][1]
        assert sent_labels == ["city", "person"]

    def test_extract_whitespace_only_labels_rejected(self, client):
        resp = client.post(
            "/extract", json={"text": "hello", "labels": ["  ", "\n"]}
        )
        assert resp.status_code == 422

    def test_extract_batch_whitespace_only_labels_rejected(self, client):
        resp = client.post(
            "/extract_batch", json={"texts": ["hello"], "labels": ["  ", "\n"]}
        )
        assert resp.status_code == 422


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
        assert "0 entities found from 3 chunk(s)" in caplog.text

    def test_short_text_does_not_log_multi_chunk_details(self, client, mock_model, caplog):
        with caplog.at_level(logging.INFO):
            resp = client.post("/extract", json={"text": "short text", "labels": ["person"]})

        assert resp.status_code == 200
        assert "exceeds chunk size" not in caplog.text
        assert "chunk 1/" not in caplog.text


# ---------------------------------------------------------------------------
# _predict_entities — batch API fallback
# ---------------------------------------------------------------------------

class TestPredictEntitiesFallback:
    def test_falls_back_to_predict_entities_when_batch_api_missing(self, monkeypatch):
        import app as app_module

        class OldModel:
            def predict_entities(self, text, labels, threshold=0.5):
                return [{"text": text, "label": labels[0], "start": 0, "end": len(text), "score": 0.9}]

        monkeypatch.setattr(app_module, "model", OldModel())

        result = app_module._predict_entities(["hello", "world"], ["person"], 0.5)

        assert result == [
            [{"text": "hello", "label": "person", "start": 0, "end": 5, "score": 0.9}],
            [{"text": "world", "label": "person", "start": 0, "end": 5, "score": 0.9}],
        ]

    def test_uses_batch_api_when_available(self, mock_model, monkeypatch):
        import app as app_module
        monkeypatch.setattr(app_module, "model", mock_model)

        app_module._predict_entities(["hello"], ["person"], 0.5)

        mock_model.batch_predict_entities.assert_called_once()
        mock_model.predict_entities.assert_not_called()


# ---------------------------------------------------------------------------
# Inference error handling — /extract and /extract_batch
# ---------------------------------------------------------------------------

class TestInferenceErrorHandling:
    def test_extract_returns_502_on_model_error(self, client, mock_model, caplog):
        mock_model.batch_predict_entities.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            resp = client.post("/extract", json={"text": "Angela Merkel", "labels": ["person"]})

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Model inference failed"
        assert "Model inference failed" in caplog.text

    def test_extract_batch_returns_502_on_model_error(self, client, mock_model):
        mock_model.batch_predict_entities.side_effect = RuntimeError("boom")

        resp = client.post(
            "/extract_batch", json={"texts": ["a", "b"], "labels": ["person"]}
        )

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Model inference failed"


# ---------------------------------------------------------------------------
# INFO-level request/summary logging
# ---------------------------------------------------------------------------

class TestRequestAndSummaryLogging:
    def test_extract_logs_request_summary(self, client, mock_model, caplog):
        text = "Angela Merkel visited Berlin"

        with caplog.at_level(logging.INFO):
            resp = client.post(
                "/extract",
                json={"text": text, "labels": ["person", "city", "person"]},
            )

        assert resp.status_code == 200
        assert f"extract: received text={len(text)} chars, labels=3 (2 after dedup)" in caplog.text

    def test_extract_batch_logs_request_summary(self, client, mock_model, caplog):
        with caplog.at_level(logging.INFO):
            resp = client.post(
                "/extract_batch",
                json={"texts": ["a", "b"], "labels": ["person", "person"]},
            )

        assert resp.status_code == 200
        assert "extract_batch: received 2 texts, labels=2 (1 after dedup)" in caplog.text

    def test_logs_cleaning_summary_for_single_chunk_text(self, client, mock_model, caplog):
        text = "Angela Merkel visited Berlin"

        with caplog.at_level(logging.INFO):
            resp = client.post("/extract", json={"text": text, "labels": ["person"]})

        assert resp.status_code == 200
        assert f"Text 0: {len(text)} chars -> {len(text)} chars after cleaning, 1 chunk(s)" in caplog.text

    def test_logs_empty_after_cleaning(self, client, mock_model, caplog):
        with caplog.at_level(logging.INFO):
            resp = client.post("/extract", json={"text": "   ", "labels": ["person"]})

        assert resp.status_code == 200
        assert "Text 0: 3 chars -> empty after cleaning, skipped" in caplog.text

    def test_logs_entities_found_summary_for_single_chunk(self, client, mock_model, caplog):
        mock_model.batch_predict_entities.return_value = [
            [
                {"text": "Angela Merkel", "label": "person", "start": 0, "end": 13, "score": 0.98},
                {"text": "Berlin", "label": "city", "start": 22, "end": 28, "score": 0.95},
            ],
        ]

        with caplog.at_level(logging.INFO):
            resp = client.post(
                "/extract",
                json={"text": "Angela Merkel visited Berlin", "labels": ["person", "city"]},
            )

        assert resp.status_code == 200
        assert "Text 0: 2 entities found from 1 chunk(s): {'person': 1, 'city': 1}" in caplog.text
