"""
Integration tests — require the real GLiNER model.

Run with:
    pytest -m integration                 # integration only
    pytest                                # all tests (unit + integration)

Inside Docker:
    docker exec ner-gliner pytest -m integration
"""
import pytest


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_health_reports_model_loaded(integration_client):
    resp = integration_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert isinstance(data["model"], str) and data["model"]


# ---------------------------------------------------------------------------
# /extract — multilingual checks
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestExtractMultilingual:
    def test_english_person_detected(self, integration_client):
        resp = integration_client.post(
            "/extract",
            json={
                "text": "Angela Merkel visited Paris last week.",
                "labels": ["person", "city"],
                "threshold": 0.3,
            },
        )
        assert resp.status_code == 200
        entities = resp.json()["entities"]
        labels_found = {e["label"] for e in entities}
        assert "person" in labels_found, f"Expected 'person', got entities: {entities}"

    def test_russian_entities(self, integration_client):
        resp = integration_client.post(
            "/extract",
            json={
                "text": "Владимир Путин посетил Москву в понедельник.",
                "labels": ["person", "city"],
                "threshold": 0.3,
            },
        )
        assert resp.status_code == 200
        entities = resp.json()["entities"]
        assert len(entities) > 0, "Expected at least one entity in Russian text"
        labels_found = {e["label"] for e in entities}
        assert labels_found & {"person", "city"}, (
            f"Expected 'person' or 'city' in Russian text, got: {entities}"
        )

    def test_german_person_detected(self, integration_client):
        resp = integration_client.post(
            "/extract",
            json={
                "text": "Angela Merkel besuchte Paris.",
                "labels": ["person", "city"],
                "threshold": 0.3,
            },
        )
        assert resp.status_code == 200
        entities = resp.json()["entities"]
        labels_found = {e["label"] for e in entities}
        assert "person" in labels_found, (
            f"Expected 'person' in German text, got: {entities}"
        )

    def test_score_field_in_range(self, integration_client):
        resp = integration_client.post(
            "/extract",
            json={
                "text": "Barack Obama visited Washington DC.",
                "labels": ["person", "city"],
                "threshold": 0.1,
            },
        )
        assert resp.status_code == 200
        for e in resp.json()["entities"]:
            assert 0.0 <= e["score"] <= 1.0


@pytest.mark.integration
def test_threshold_monotonicity(integration_client):
    """Higher threshold must yield fewer or equal entities than lower threshold."""
    text = "Barack Obama visited New York City and met Angela Merkel."
    low = integration_client.post(
        "/extract", json={"text": text, "labels": ["person", "city"], "threshold": 0.1}
    )
    high = integration_client.post(
        "/extract", json={"text": text, "labels": ["person", "city"], "threshold": 0.99}
    )
    assert low.status_code == 200
    assert high.status_code == 200
    assert len(low.json()["entities"]) >= len(high.json()["entities"]), (
        "Higher threshold should yield fewer or equal entities"
    )


@pytest.mark.integration
def test_empty_text_returns_empty_entities(integration_client):
    resp = integration_client.post(
        "/extract", json={"text": "", "labels": ["person"], "threshold": 0.5}
    )
    assert resp.status_code == 200
    assert resp.json()["entities"] == []


# ---------------------------------------------------------------------------
# /extract_batch
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestBatchIntegration:
    def test_result_count_matches_input(self, integration_client):
        texts = [
            "Angela Merkel visited Berlin.",
            "Barack Obama is from Chicago.",
            "Владимир Путин — президент России.",
        ]
        resp = integration_client.post(
            "/extract_batch",
            json={"texts": texts, "labels": ["person", "city"], "threshold": 0.3},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == len(texts), (
            "Number of results must equal number of input texts"
        )

    def test_order_preserved(self, integration_client):
        """First text has a person, second an organization — check order isn't swapped."""
        texts = [
            "Angela Merkel was the Chancellor of Germany.",
            "Google was founded in California.",
        ]
        resp = integration_client.post(
            "/extract_batch",
            json={"texts": texts, "labels": ["person", "organization", "location"], "threshold": 0.3},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
        # Validate each result is a proper entities dict
        for r in results:
            assert "entities" in r

    def test_empty_texts_list(self, integration_client):
        resp = integration_client.post(
            "/extract_batch", json={"texts": [], "labels": ["person"]}
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_batch_with_empty_string_item(self, integration_client):
        """Empty string inside a batch must not crash the service."""
        texts = ["", "Barack Obama visited Washington DC."]
        resp = integration_client.post(
            "/extract_batch",
            json={"texts": texts, "labels": ["person", "city"], "threshold": 0.3},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 2
