# GLiNER NER Service

REST-сервис на FastAPI с моделью [urchade/gliner_multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1).  
Принимает произвольный текст и список типов сущностей (задаются клиентом), возвращает извлечённые сущности в JSON.  
Образ переносится через `docker save` / `docker load` и работает **без доступа в интернет**.

---

## Лицензия модели — важно прочитать

Модель `urchade/gliner_multi-v2.1` распространяется под лицензией **Creative Commons Attribution Non-Commercial 4.0 (CC BY-NC 4.0)**.

> **Коммерческое использование запрещено.**  
> Если вам нужен вариант для коммерции, рассмотрите другие чекпойнты GLiNER,
> распространяемые под Apache 2.0 (например `urchade/gliner_large-v2.1` — проверьте
> актуальную лицензию на странице модели перед использованием).

---

## Структура проекта

```
ner-gliner/
├── app.py                # FastAPI-сервис
├── requirements.txt      # зависимости (без torch — устанавливается отдельно)
├── requirements-dev.txt  # pytest, httpx
├── Dockerfile
├── .dockerignore
├── README.md
└── tests/
    ├── conftest.py       # фикстуры, маркер integration
    ├── test_unit.py      # быстрые тесты с моком
    └── test_integration.py  # медленные тесты с реальной моделью
```

---

## Сборка образа

```bash
docker build --platform linux/amd64 -t ner-gliner:1.0 .
```

На этапе сборки:
1. Устанавливается CPU-вариант torch (без CUDA-библиотек).
2. Скачивается модель `urchade/gliner_multi-v2.1` и «запекается» в образ.
3. Прогоняется smoke-тест — если модель не детектирует `person`, сборка падает.

После сборки переменные `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`
гарантируют, что в рантайме контейнер **не обращается к HuggingFace Hub**.

---

## Перенос на офлайн-машину

```bash
# На машине со сборкой
docker save ner-gliner:1.0 | gzip > ner-gliner.tar.gz

# На целевой машине
gunzip -c ner-gliner.tar.gz | docker load
```

---

## Запуск

```bash
docker run -d -p 8000:8000 --name ner-gliner ner-gliner:1.0
```

Проверка без сети:

```bash
docker run --rm --network none -p 8000:8000 ner-gliner:1.0
```

### Параметр числа воркеров

Каждый воркер uvicorn держит **отдельную копию модели** (~1.5 GB RAM каждая).
По умолчанию `WORKERS=1`.

```bash
docker run -d -p 8000:8000 -e WORKERS=2 --name ner-gliner ner-gliner:1.0
```

---

## API

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model": "urchade/gliner_multi-v2.1", "model_loaded": true}
```

---

### POST /extract

Один текст, произвольный набор меток.

**Запрос:**

```json
{
  "text": "Angela Merkel besuchte Paris.",
  "labels": ["person", "city"],
  "threshold": 0.5
}
```

**Ответ:**

```json
{
  "entities": [
    {"text": "Angela Merkel", "label": "person", "start": 0,  "end": 13, "score": 0.984},
    {"text": "Paris",         "label": "city",   "start": 22, "end": 27, "score": 0.961}
  ]
}
```

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"Angela Merkel besuchte Paris.","labels":["person","city"]}'
```

Многоязычный пример (русский):

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"Владимир Путин посетил Москву.","labels":["person","city"],"threshold":0.3}'
```

---

### POST /extract_batch

Список текстов одним запросом с общим набором меток.  
Порядок элементов в `results` строго соответствует порядку `texts`.

**Запрос:**

```json
{
  "texts": [
    "Angela Merkel visited Berlin.",
    "Barack Obama is from Chicago."
  ],
  "labels": ["person", "city"],
  "threshold": 0.5
}
```

**Ответ:**

```json
{
  "results": [
    {"entities": [{"text": "Angela Merkel", "label": "person", "start": 0, "end": 13, "score": 0.98},
                  {"text": "Berlin",        "label": "city",   "start": 22,"end": 28, "score": 0.95}]},
    {"entities": [{"text": "Barack Obama", "label": "person", "start": 0, "end": 12, "score": 0.97},
                  {"text": "Chicago",      "label": "city",   "start": 21,"end": 28, "score": 0.93}]}
  ]
}
```

```bash
curl -X POST http://localhost:8000/extract_batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Angela Merkel visited Berlin.","Barack Obama is from Chicago."],"labels":["person","city"]}'
```

---

## Контракт — важные детали

| Тема | Детали |
|---|---|
| **labels** | Обязательный непустой список; задаётся клиентом в каждом запросе. |
| **Язык меток** | Метки лучше задавать **по-английски** — модель так точнее и стабильнее. |
| **threshold** | Опционально, по умолчанию `0.5`. Диапазон `[0.0, 1.0]`. Меньше → больше сущностей, выше recall. |
| **score** | Вероятность [0, 1] с точностью до 4 знаков. |
| **start / end** | Позиции в **Unicode code points** (не байтах). В Java — `text.codePoints()` или `text.offsetByCodePoints()`. |
| **Скорость** | Латентность линейно растёт с длиной текста и числом меток. Используйте `/extract_batch` вместо цикла по одному тексту. |
| **Память** | Каждый воркер uvicorn грузит свою копию модели (~1.5 GB). При `WORKERS=1` (дефолт) — одна копия. |

---

## Тесты

### Зависимости для тестов

```bash
# Только FastAPI/Pydantic нужны для юнит-тестов (torch/gliner не требуется)
pip install fastapi pydantic uvicorn httpx pytest

# Для интеграционных тестов — полный стек:
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Запуск

```bash
# Только быстрые юниты (мок модели, секунды, без torch)
pytest -m "not integration"

# Только интеграционные (реальная модель, нужны веса)
pytest -m integration

# Всё вместе
pytest
```

### Запуск внутри контейнера

```bash
# Установить dev-зависимости внутри контейнера
docker exec ner-gliner pip install pytest httpx

# Интеграционные тесты с реальной загруженной моделью
docker exec ner-gliner pytest -m integration
```

> **Примечание:** интеграционные тесты требуют наличия весов модели —
> либо внутри Docker-образа, либо локально после первой загрузки.
