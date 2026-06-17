# syntax=docker/dockerfile:1.4
FROM --platform=linux/amd64 python:3.10.11-slim

WORKDIR /app

# Some HF tokenizer packages need git at download time
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so transitive deps pick it up instead of CUDA variant
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# Bake model weights into the image.
# Running both download AND a smoke-test here means a bad checkpoint
# fails at build time, not at container start.
# --------------------------------------------------------------------------
RUN <<'PYEOF'
#!/usr/bin/env python
import sys, time
from gliner import GLiNER

MODEL = "urchade/gliner_multi-v2.1"
print(f"Downloading {MODEL} ...", flush=True)
t0 = time.time()
m = GLiNER.from_pretrained(MODEL)
print(f"Downloaded in {time.time() - t0:.1f} s", flush=True)

ents = m.predict_entities(
    "Angela Merkel visited Paris last Tuesday.",
    ["person", "city", "date"],
    threshold=0.3,
)
labels = {e["label"] for e in ents}
print("Smoke-test entities:", ents, flush=True)
if "person" not in labels:
    sys.exit(f"Smoke-test FAILED — expected label 'person', got: {labels}")
print("Smoke-test passed.", flush=True)
PYEOF

# Install monitoring dependencies in their own layer so changes here don't
# invalidate the cached model download above.
COPY requirements-monitoring.txt .
RUN pip install --no-cache-dir -r requirements-monitoring.txt

# Copy application and tests (tests allow running pytest inside the container)
COPY app.py .
COPY tests/ tests/

# --------------------------------------------------------------------------
# Offline mode: prevent any HF network calls at runtime.
# These ENV vars are set AFTER all download RUN steps above.
# --------------------------------------------------------------------------
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    WORKERS=1

EXPOSE 8000

# Give the model ~2 min to load before the first health check fires
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
        || exit 1

# WORKERS can be overridden at runtime: docker run -e WORKERS=2 ...
# Note: each uvicorn worker loads its own model copy — memory scales linearly.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
