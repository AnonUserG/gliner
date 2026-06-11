import os
import re
import time
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _env_int(name: str) -> Optional[int]:
    """Read an optional positive-int env var; unset/empty means unlimited."""
    value = os.environ.get(name)
    return int(value) if value else None


MODEL_NAME = os.environ.get("MODEL_NAME", "urchade/gliner_multi-v2.1")
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.5"))

# Optional limits — unset/empty env var means unlimited. Configurable at
# container start so the image rarely needs rebuilding.
MAX_TEXT_LENGTH = _env_int("MAX_TEXT_LENGTH")
MAX_BATCH_SIZE = _env_int("MAX_BATCH_SIZE")

# GLiNER's underlying span model truncates each input to `config.max_len`
# "words" (default 384), silently dropping anything beyond that — see
# gliner/data_processing/processor.py. To support arbitrarily long texts
# without losing entities, long texts are split into word-chunks and the
# per-chunk results are merged back using the original character offsets.
CHUNK_SIZE_WORDS = int(os.environ.get("CHUNK_SIZE_WORDS", "300"))

# Mirrors gliner's default WhitespaceTokenSplitter so a "word" here lines up
# with what the model counts towards `config.max_len`.
_WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")

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
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float


class ExtractResponse(BaseModel):
    entities: list[Entity]


class ExtractBatchRequest(BaseModel):
    texts: list[Annotated[str, Field(max_length=MAX_TEXT_LENGTH)]] = Field(
        ..., max_length=MAX_BATCH_SIZE
    )
    labels: list[str] = Field(..., min_length=1)
    threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0.0, le=1.0)


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


def _chunk_text(text: str, chunk_size: int) -> list[tuple[int, int]]:
    """Split `text` into chunks of up to `chunk_size` words.

    Returns a list of (start, end) character offsets into `text`, one per
    chunk. Chunks are non-overlapping and cover the whole text; a chunk's
    substring is `text[start:end]`.
    """
    words = list(_WORD_PATTERN.finditer(text))
    if not words:
        return [(0, len(text))]

    chunks = []
    for i in range(0, len(words), chunk_size):
        group = words[i : i + chunk_size]
        chunks.append((group[0].start(), group[-1].end()))
    return chunks


def _predict_entities(texts: list[str], labels: list[str], threshold: float) -> list[list[dict]]:
    try:
        return model.batch_predict_entities(texts, labels, threshold=threshold)
    except AttributeError:
        # fallback for older gliner versions without batch API
        return [model.predict_entities(text, labels, threshold=threshold) for text in texts]


def _extract_entities(texts: list[str], labels: list[str], threshold: float) -> list[list[dict]]:
    """Extract entities for each text, transparently chunking long texts.

    Texts longer than CHUNK_SIZE_WORDS words are split into chunks, each
    chunk is run through the model independently, and the resulting
    entities are merged back with their `start`/`end` offsets translated to
    positions in the original text. All chunks across all input texts are
    sent to the model in a single batched call.
    """
    chunk_texts: list[str] = []
    # one entry per chunk, in the same order as chunk_texts
    chunk_owner: list[tuple[int, int]] = []  # (text_index, char_offset)
    chunk_counts: list[int] = [0] * len(texts)

    for text_idx, text in enumerate(texts):
        if not text:
            continue
        spans = _chunk_text(text, CHUNK_SIZE_WORDS)
        chunk_counts[text_idx] = len(spans)
        if len(spans) > 1:
            word_count = len(_WORD_PATTERN.findall(text))
            logger.info(
                "Text %d: %d words (%d chars) exceeds chunk size %d — split into %d chunks",
                text_idx, word_count, len(text), CHUNK_SIZE_WORDS, len(spans),
            )
        for start, end in spans:
            chunk_texts.append(text[start:end])
            chunk_owner.append((text_idx, start))

    results: list[list[dict]] = [[] for _ in texts]
    if not chunk_texts:
        return results

    raw_chunks = _predict_entities(chunk_texts, labels, threshold)

    chunk_seen = [0] * len(texts)
    for (text_idx, offset), chunk_text, chunk_entities in zip(chunk_owner, chunk_texts, raw_chunks):
        for entity in chunk_entities:
            entity["start"] += offset
            entity["end"] += offset
        results[text_idx].extend(chunk_entities)

        if chunk_counts[text_idx] > 1:
            chunk_seen[text_idx] += 1
            logger.info(
                "Text %d chunk %d/%d (offset %d, %d chars): extracted %d entities",
                text_idx, chunk_seen[text_idx], chunk_counts[text_idx],
                offset, len(chunk_text), len(chunk_entities),
            )

    for text_idx, count in enumerate(chunk_counts):
        if count > 1:
            logger.info(
                "Text %d: merged %d entities from %d chunks",
                text_idx, len(results[text_idx]), count,
            )

    return results


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
    raw = _extract_entities([req.text], req.labels, req.threshold)[0]
    logger.debug("Inference /extract: %.3f s, %d entities", time.time() - t0, len(raw))

    return ExtractResponse(entities=_format_entities(raw))


@app.post("/extract_batch", response_model=ExtractBatchResponse)
def extract_batch(req: ExtractBatchRequest):
    if not req.texts:
        return ExtractBatchResponse(results=[])

    t0 = time.time()
    raw_batch = _extract_entities(req.texts, req.labels, req.threshold)
    logger.debug(
        "Inference /extract_batch: %.3f s, %d texts",
        time.time() - t0,
        len(req.texts),
    )

    return ExtractBatchResponse(
        results=[ExtractResponse(entities=_format_entities(r)) for r in raw_batch]
    )
