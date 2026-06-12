import html
import os
import re
import time
import logging
import unicodedata
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Annotated, Any, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from opentelemetry import trace
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Per-request correlation ID, set by the middleware below and injected into
# every log line so concurrent requests can be told apart. "-" for log
# records emitted outside a request (e.g. at startup).
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


class _TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get()
        return True


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s [%(trace_id)s] - %(message)s",
)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_TraceIdFilter())

logger = logging.getLogger(__name__)


class _AccessLogFilter(logging.Filter):
    """Drops uvicorn access-log lines for endpoints polled on a timer
    (Prometheus scraping /metrics, Docker healthchecks hitting /health),
    which would otherwise dominate the logs.
    """

    _IGNORED_PATHS = ("/metrics", "/health")

    def filter(self, record: logging.LogRecord) -> bool:
        if len(record.args) >= 3:
            path = record.args[2].split("?", 1)[0]
            return path not in self._IGNORED_PATHS
        return True


logging.getLogger("uvicorn.access").addFilter(_AccessLogFilter())


def _env_int(name: str) -> Optional[int]:
    """Read an optional positive-int env var; unset/empty means unlimited."""
    value = os.environ.get(name)
    return int(value) if value else None


MODEL_NAME = os.environ.get("MODEL_NAME", "urchade/gliner_multi-v2.1")
DEFAULT_THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.5"))

# Zipkin collector endpoint for trace export, e.g. http://zipkin:9411/api/v2/spans.
# Unset/empty disables tracing entirely.
ZIPKIN_ENDPOINT = os.environ.get("ZIPKIN_ENDPOINT")
OTEL_SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "ner-gliner")

# Optional limits — unset/empty env var means unlimited. Configurable at
# container start so the image rarely needs rebuilding.
MAX_TEXT_LENGTH = _env_int("MAX_TEXT_LENGTH")
MAX_BATCH_SIZE = _env_int("MAX_BATCH_SIZE")
MAX_LABELS = _env_int("MAX_LABELS")

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
# Monitoring: Prometheus metrics + Zipkin tracing
# ---------------------------------------------------------------------------

# Exposes GET /metrics in Prometheus text format (request counts, latency
# histograms, request/response sizes, etc.).
Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def _add_trace_id(request: Request, call_next):
    """Assign a trace ID to the request and make it available to logging.

    Reuses the active OpenTelemetry span's trace ID when Zipkin tracing is
    enabled (so logs can be correlated with Zipkin traces), otherwise
    generates a random one. Registered before FastAPIInstrumentor.instrument_app()
    below so that, when tracing is enabled, the OTel span is already current
    by the time this middleware runs.
    """
    span_context = trace.get_current_span().get_span_context()
    trace_id = (
        format(span_context.trace_id, "032x")
        if span_context.is_valid
        else uuid.uuid4().hex
    )
    token = _trace_id_var.set(trace_id)
    try:
        return await call_next(request)
    finally:
        _trace_id_var.reset(token)


class _QuietZipkinExporter(ZipkinExporter):
    """ZipkinExporter that reports connectivity loss/recovery only once.

    By default, a failed export raises and BatchSpanProcessor logs a full
    ERROR traceback for every batch — every few seconds while Zipkin is
    down. This wraps export() so the failure is logged once, stays quiet
    on repeated failures, and logs once more when exports start succeeding
    again.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unreachable = False

    def export(self, spans):
        try:
            result = super().export(spans)
        except requests.exceptions.RequestException as exc:
            if not self._unreachable:
                logger.warning(
                    "Zipkin (%s) unreachable, dropping traces until it recovers: %s",
                    self.endpoint, exc,
                )
                self._unreachable = True
            return SpanExportResult.FAILURE

        if result == SpanExportResult.SUCCESS:
            if self._unreachable:
                logger.info("Zipkin (%s) reachable again, resuming trace export", self.endpoint)
                self._unreachable = False
        elif not self._unreachable:
            logger.warning("Zipkin (%s) unreachable, dropping traces until it recovers", self.endpoint)
            self._unreachable = True

        return result


if ZIPKIN_ENDPOINT:
    _tracer_provider = TracerProvider(
        resource=Resource.create({"service.name": OTEL_SERVICE_NAME})
    )
    _tracer_provider.add_span_processor(
        BatchSpanProcessor(_QuietZipkinExporter(endpoint=ZIPKIN_ENDPOINT))
    )
    trace.set_tracer_provider(_tracer_provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("Zipkin tracing enabled, sending spans to %s", ZIPKIN_ENDPOINT)
else:
    logger.info("Zipkin tracing disabled (ZIPKIN_ENDPOINT not set)")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ExtractRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)
    labels: list[str] = Field(..., min_length=1, max_length=MAX_LABELS)
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
    labels: list[str] = Field(..., min_length=1, max_length=MAX_LABELS)
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


# Characters that carry no semantic meaning but routinely show up in
# copy-pasted or scraped text: C0/C1 control codes (keeping \t \n \r, which
# the whitespace pass below collapses anyway), zero-width spaces/joiners,
# the word joiner, BOM, and soft hyphen.
_INVISIBLE_CHAR_PATTERN = re.compile(
    "[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f"
    "\\u00ad\\u200b-\\u200d\\u2060\\ufeff]"
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Strip markup/invisible characters and normalize whitespace.

    Applied to every input text before chunking and inference, so the
    `start`/`end` offsets in the response refer to positions in this
    cleaned text, not the raw input.

    Order matters: HTML entities are decoded first so any tags they spell
    out (e.g. "&lt;b&gt;") get stripped too, then NFKC normalization folds
    full-width/compatibility characters (including turning NBSP and other
    Unicode spaces into a regular space) before remaining invisible
    characters are removed and whitespace is collapsed.
    """
    text = html.unescape(text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_CHAR_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _clean_labels(labels: list[str]) -> list[str]:
    """Strip whitespace, drop empty entries, and remove duplicate labels.

    The order of first occurrence is preserved.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for label in labels:
        label = label.strip()
        if not label or label in seen:
            continue
        seen.add(label)
        cleaned.append(label)
    return cleaned


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
    if hasattr(model, "batch_predict_entities"):
        return model.batch_predict_entities(texts, labels, threshold=threshold)
    # fallback for older gliner versions without batch API
    return [model.predict_entities(text, labels, threshold=threshold) for text in texts]


def _extract_entities(texts: list[str], labels: list[str], threshold: float) -> list[list[dict]]:
    """Extract entities for each text, transparently chunking long texts.

    Each text is first run through `_clean_text`, so `start`/`end` offsets
    below — and in the response — refer to positions in the cleaned text,
    not the raw input. Texts longer than CHUNK_SIZE_WORDS words are then
    split into chunks, each chunk is run through the model independently,
    and the resulting entities are merged back with their `start`/`end`
    offsets translated to positions in the cleaned text. All chunks across
    all input texts are sent to the model in a single batched call.
    """
    chunk_texts: list[str] = []
    # one entry per chunk, in the same order as chunk_texts
    chunk_owner: list[tuple[int, int]] = []  # (text_index, char_offset)
    chunk_counts: list[int] = [0] * len(texts)

    for text_idx, text in enumerate(texts):
        raw_len = len(text)
        text = _clean_text(text)
        if not text:
            logger.info("Text %d: %d chars -> empty after cleaning, skipped", text_idx, raw_len)
            continue

        spans = _chunk_text(text, CHUNK_SIZE_WORDS)
        chunk_counts[text_idx] = len(spans)
        logger.info(
            "Text %d: %d chars -> %d chars after cleaning, %d chunk(s)",
            text_idx, raw_len, len(text), len(spans),
        )
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

    try:
        raw_chunks = _predict_entities(chunk_texts, labels, threshold)
    except Exception:
        logger.exception(
            "Model inference failed for %d chunk(s), %d label(s), threshold=%.2f",
            len(chunk_texts), len(labels), threshold,
        )
        raise HTTPException(502, detail="Model inference failed")

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

    for text_idx, entities in enumerate(results):
        if chunk_counts[text_idx] == 0:
            continue
        label_counts = dict(Counter(e["label"] for e in entities))
        logger.info(
            "Text %d: %d entities found from %d chunk(s): %s",
            text_idx, len(entities), chunk_counts[text_idx], label_counts,
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
    labels = _clean_labels(req.labels)
    logger.info(
        "extract: received text=%d chars, labels=%d (%d after dedup)",
        len(req.text), len(req.labels), len(labels),
    )
    if not labels:
        raise HTTPException(422, detail="labels must contain at least one non-empty value")

    if not req.text:
        return ExtractResponse(entities=[])

    t0 = time.time()
    raw = _extract_entities([req.text], labels, req.threshold)[0]
    logger.debug("Inference /extract: %.3f s, %d entities", time.time() - t0, len(raw))

    return ExtractResponse(entities=_format_entities(raw))


@app.post("/extract_batch", response_model=ExtractBatchResponse)
def extract_batch(req: ExtractBatchRequest):
    labels = _clean_labels(req.labels)
    logger.info(
        "extract_batch: received %d texts, labels=%d (%d after dedup)",
        len(req.texts), len(req.labels), len(labels),
    )
    if not labels:
        raise HTTPException(422, detail="labels must contain at least one non-empty value")

    if not req.texts:
        return ExtractBatchResponse(results=[])

    t0 = time.time()
    raw_batch = _extract_entities(req.texts, labels, req.threshold)
    logger.debug(
        "Inference /extract_batch: %.3f s, %d texts",
        time.time() - t0,
        len(req.texts),
    )

    return ExtractBatchResponse(
        results=[ExtractResponse(entities=_format_entities(r)) for r in raw_batch]
    )
