import time
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "urchade/gliner_multi-v2.1"
MAX_TEXT_LENGTH = 10_000  # characters; protects against OOM on very long inputs

model: Optional[Any] = None
model_loaded: bool = False


def _load_model() -> Any:
    from gliner import GLiNER  # lazy import — keeps unit tests free of torch/gliner
    return GLiNER.from_pretrained(MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_loaded
    logger.info("Loading GLiNER model: %s", MODEL_NAME)
    t0 = time.time()
    model = _load_model()
    model_loaded = True
    logger.info("Model loaded in %.2f s", time.time() - t0)
    yield
    model = None
    model_loaded = False


app = FastAPI(title="GLiNER NER Service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    labels: list[str] = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float


class ExtractResponse(BaseModel):
    entities: list[Entity]


class ExtractBatchRequest(BaseModel):
    texts: list[Annotated[str, Field(max_length=MAX_TEXT_LENGTH)]]
    labels: list[str] = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractBatchResponse(BaseModel):
    results: list[ExtractResponse]


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_entities(raw: list[dict]) -> list[Entity]:
    return [
        Entity(
            text=e["text"],
            label=e["label"],
            start=e["start"],
            end=e["end"],
            score=round(float(e["score"]), 4),
        )
        for e in raw
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model=MODEL_NAME, model_loaded=model_loaded)


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest):
    if not req.text:
        return ExtractResponse(entities=[])

    t0 = time.time()
    raw = model.predict_entities(req.text, req.labels, threshold=req.threshold)
    logger.debug("Inference /extract: %.3f s, %d entities", time.time() - t0, len(raw))

    return ExtractResponse(entities=_format_entities(raw))


@app.post("/extract_batch", response_model=ExtractBatchResponse)
def extract_batch(req: ExtractBatchRequest):
    if not req.texts:
        return ExtractBatchResponse(results=[])

    t0 = time.time()
    try:
        raw_batch = model.batch_predict_entities(
            req.texts, req.labels, threshold=req.threshold
        )
    except AttributeError:
        # fallback for older gliner versions without batch API
        raw_batch = [
            model.predict_entities(text, req.labels, threshold=req.threshold)
            for text in req.texts
        ]
    logger.debug(
        "Inference /extract_batch: %.3f s, %d texts",
        time.time() - t0,
        len(req.texts),
    )

    return ExtractBatchResponse(
        results=[ExtractResponse(entities=_format_entities(r)) for r in raw_batch]
    )
