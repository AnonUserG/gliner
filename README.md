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
docker save ner-gliner:1.0 -o ner-gliner.tar

# На целевой машине
docker load -i ner-gliner.tar
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

### Переменные окружения

Все параметры ниже задаются через `-e` при запуске контейнера — пересборка образа не требуется.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `WORKERS` | `1` | Число воркеров uvicorn. Каждый держит **отдельную копию модели** (~1.5 GB RAM). |
| `MODEL_NAME` | `urchade/gliner_multi-v2.1` | Имя модели для `GLiNER.from_pretrained`. При смене модели на офлайн-машине (`HF_HUB_OFFLINE=1`) веса должны быть уже доступны (запечены в образ или закэшированы). |
| `MAX_TEXT_LENGTH` | не задано (без ограничений) | Максимальная длина `text` / элемента `texts` в символах. Запрос с более длинным текстом вернёт `422`. |
| `MAX_BATCH_SIZE` | не задано (без ограничений) | Максимальное число элементов в `texts` для `/extract_batch`. |
| `MAX_LABELS` | не задано (без ограничений) | Максимальное число элементов в `labels` (до очистки/дедупликации — см. раздел "Контракт"). Запрос с большим количеством меток вернёт `422`. |
| `DEFAULT_THRESHOLD` | `0.5` | Значение `threshold`, если оно не передано в запросе. |
| `LOG_LEVEL` | `INFO` | Уровень логирования (`DEBUG`, `INFO`, `WARNING`, ...). |
| `CHUNK_SIZE_WORDS` | `300` | Размер чанка (в словах) для разбиения длинных текстов — см. раздел "Чанкование длинных текстов". |

```bash
docker run -d -p 8000:8000 \
  -e WORKERS=2 \
  -e MAX_TEXT_LENGTH=10000 \
  -e MAX_BATCH_SIZE=100 \
  -e DEFAULT_THRESHOLD=0.3 \
  --name ner-gliner ner-gliner:1.0
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

## Примеры меток (labels)

Модель zero-shot — метки задаются произвольно на английском. Ниже распространённые наборы:

| Категория | Пример labels |
|---|---|
| Люди и места | `["person", "city", "country"]` |
| Организации | `["person", "organization", "location"]` |
| Время и события | `["person", "date", "event"]` |
| Бизнес и финансы | `["company", "product", "currency", "price"]` |
| Медицина | `["disease", "drug", "symptom", "doctor"]` |
| Новости / СМИ | `["person", "organization", "location", "date", "phone number", "email"]` |
| Вакансии / резюме | `["person", "job title", "company", "skill", "city"]` |
| Юридические тексты | `["person", "organization", "law", "date", "court"]` |
| Недвижимость | `["property type", "location", "price", "area", "developer"]` |
| Спорт | `["athlete", "team", "sport", "tournament", "score"]` |
| Логистика | `["sender", "recipient", "address", "tracking number", "cargo"]` |
| Модерация контента | `["insult", "threat", "profanity", "hate speech"]` |

> Метки лучше задавать **по-английски** — модель точнее и стабильнее с английскими токенами меток, даже если сам текст на другом языке.
>
> **Важно:** GLiNER извлекает конкретные *фрагменты* текста. Метки вида `"негативный текст"` или `"токсичный отзыв"` работают плохо — это характеристика всего текста, а не выделяемого span'а. Для классификации тональности используйте отдельные модели. Вместо этого используйте конкретные метки: `"insult"`, `"threat"`, `"profanity"`.

---

### Пример: мониторинг СМИ

Извлечение персон, локаций, организаций, контактов и упоминаний конкретной компании (здесь — `"Газпром"`).  
Название нужной организации передаётся как отдельная метка — модель найдёт только её упоминания.

```json
{
  "text": "Представитель Газпрома Иван Петров сообщил, что компания открыла офис в Екатеринбурге. По вопросам сотрудничества: +7 (495) 719-30-01, press@gazprom.ru",
  "labels": ["person", "organization", "location", "phone number", "email", "Газпром"],
  "threshold": 0.3
}
```

Ожидаемый результат:

```json
{
  "entities": [
    {"text": "Газпрома",          "label": "Газпром",      "start": 14,  "end": 22,  "score": 0.91},
    {"text": "Иван Петров",       "label": "person",       "start": 23,  "end": 34,  "score": 0.97},
    {"text": "Екатеринбурге",     "label": "location",     "start": 71,  "end": 84,  "score": 0.95},
    {"text": "+7 (495) 719-30-01","label": "phone number", "start": 114, "end": 132, "score": 0.88},
    {"text": "press@gazprom.ru",  "label": "email",        "start": 134, "end": 150, "score": 0.93}
  ]
}
```

> **Метка = название организации** — удобно когда нужно отслеживать упоминания конкретного бренда или компании отдельно от всех остальных организаций в тексте.

---

### Пример: модерация контента

Поиск конкретных токсичных фраз и угроз в пользовательских сообщениях.

```json
{
  "text": "Ты полный идиот, я тебя найду и тебе не поздоровится. Убирайся отсюда.",
  "labels": ["insult", "threat", "profanity"],
  "threshold": 0.3
}
```

Ожидаемый результат:

```json
{
  "entities": [
    {"text": "полный идиот",                    "label": "insult",   "start": 3,  "end": 15, "score": 0.89},
    {"text": "я тебя найду и тебе не поздоровится", "label": "threat", "start": 17, "end": 52, "score": 0.84}
  ]
}
```

---

### Пример: недвижимость

Извлечение параметров объекта из текста объявления.

```json
{
  "text": "Продаётся 2-комнатная квартира 58 кв.м в ЖК Олимп, Казань, застройщик ПИК. Цена 4 500 000 руб.",
  "labels": ["property type", "area", "location", "developer", "price"],
  "threshold": 0.3
}
```

Ожидаемый результат:

```json
{
  "entities": [
    {"text": "2-комнатная квартира", "label": "property type", "start": 10, "end": 30, "score": 0.93},
    {"text": "58 кв.м",             "label": "area",          "start": 31, "end": 38, "score": 0.91},
    {"text": "Казань",              "label": "location",      "start": 51, "end": 57, "score": 0.96},
    {"text": "ПИК",                 "label": "developer",     "start": 71, "end": 74, "score": 0.88},
    {"text": "4 500 000 руб.",      "label": "price",         "start": 82, "end": 96, "score": 0.90}
  ]
}
```

---

### Пример: вакансии и резюме

Парсинг ключевых данных из текста резюме или объявления о вакансии.

```json
{
  "text": "Иван Сидоров, senior Python developer, 8 лет опыта. Работал в Яндексе и Сбере. Ищу позицию в Москве или удалённо.",
  "labels": ["person", "job title", "skill", "company", "location"],
  "threshold": 0.3
}
```

Ожидаемый результат:

```json
{
  "entities": [
    {"text": "Иван Сидоров",          "label": "person",    "start": 0,  "end": 12, "score": 0.97},
    {"text": "senior Python developer","label": "job title", "start": 14, "end": 36, "score": 0.92},
    {"text": "Python",                "label": "skill",     "start": 21, "end": 27, "score": 0.88},
    {"text": "Яндексе",               "label": "company",   "start": 62, "end": 69, "score": 0.94},
    {"text": "Сбере",                 "label": "company",   "start": 72, "end": 77, "score": 0.91},
    {"text": "Москве",                "label": "location",  "start": 95, "end": 101,"score": 0.95}
  ]
}
```

---

## Контракт — важные детали

| Тема | Детали |
|---|---|
| **labels** | Обязательный непустой список; задаётся клиентом в каждом запросе. Перед обработкой у каждой метки обрезаются пробелы по краям, пустые после обрезки метки убираются, дубликаты схлопываются (остаётся первое вхождение). Если после очистки список пуст — `422`. Ограничение `MAX_LABELS` (если задано) применяется к исходному списку, до этой очистки. |
| **Язык меток** | Метки лучше задавать **по-английски** — модель так точнее и стабильнее. |
| **threshold** | Опционально, по умолчанию `0.5`. Диапазон `[0.0, 1.0]`. Меньше → больше сущностей, выше recall. |
| **score** | Вероятность [0, 1] с точностью до 4 знаков. |
| **start / end** | Позиции в **Unicode code points** (не байтах) **в очищенном тексте** — см. раздел "Очистка входного текста". В Java — `text.codePoints()` или `text.offsetByCodePoints()`. |
| **Скорость** | Латентность линейно растёт с длиной текста и числом меток. Используйте `/extract_batch` вместо цикла по одному тексту. |
| **Ошибки модели** | Если сам инференс GLiNER завершился с ошибкой (например, нехватка памяти), эндпоинт вернёт `502 {"detail": "Model inference failed"}`. Подробности — в логах сервиса (уровень `ERROR`). |
| **Память** | Каждый воркер uvicorn грузит свою копию модели (~1.5 GB). При `WORKERS=1` (дефолт) — одна копия. |

---

## Очистка входного текста

Перед извлечением сущностей каждый текст (и каждый элемент `texts` в `/extract_batch`) автоматически очищается:

- **HTML-теги** удаляются (заменяются пробелом, чтобы соседние слова не склеились), **HTML-сущности** декодируются (`&amp;` → `&`, `&nbsp;` → пробел и т.п.).
- **Невидимые и управляющие символы** убираются: control-символы (кроме `\n`, `\t`, `\r`), zero-width space/joiner, word joiner, BOM, мягкий дефис.
- **Unicode-нормализация (NFKC)**: полноширинные символы, неразрывные и другие нестандартные пробелы и т.п. приводятся к каноническому виду (например, NBSP становится обычным пробелом).
- **Повторяющиеся пробелы/табы/переводы строк** схлопываются в один пробел, начальные и конечные пробелы убираются.

> **Важно:** `start`/`end` в ответе указывают на позиции в **очищенном** тексте, а не в исходном `text`/`texts`, присланном клиентом. Если входной текст не содержал HTML/спецсимволов и не имел лишних пробелов, очистка не меняет его — позиции совпадают с исходными.
>
> Если текст после очистки оказывается пустым (например, состоял только из HTML-тегов или пробелов), для него возвращается пустой список `entities` без ошибки.

---

## Чанкование длинных текстов

Модель GLiNER внутри обрезает каждый вход до `config.max_len` слов (по умолчанию **384**), молча отбрасывая всё, что не поместилось.
Чтобы сущности из длинных текстов не терялись, сервис сам разбивает каждый текст на чанки по `CHUNK_SIZE_WORDS` слов (по умолчанию **300**), прогоняет каждый чанк через модель и склеивает результаты обратно в один список сущностей с пересчётом `start`/`end` в координаты исходного текста.

- Чанки **не пересекаются** и режутся по границам слов (тот же сплиттер, что использует GLiNER по умолчанию — `\w+(?:[-_]\w+)*|\S`).
- Сущность, которая физически разорвана между двумя чанками (например, "New" в конце одного чанка и "York" в начале следующего), может быть найдена не полностью или не найдена вовсе — это известное ограничение текущей реализации.
- Чанки всех текстов одного запроса `/extract_batch` отправляются в модель **одним батч-вызовом**.
- Если текст разбит на несколько чанков, в логи (уровень `INFO`) дополнительно пишется прогресс по каждому чанку и итог склейки — см. раздел "Логирование".

---

## Логирование

Сервис пишет логи в консоль (stdout) — удобно смотреть через `docker logs`. Уровень задаётся переменной `LOG_LEVEL` (по умолчанию `INFO`).

На уровне `INFO` для каждого запроса к `/extract` и `/extract_batch` пишется:

- **Сводка запроса**: длина текста (или количество текстов для батча) и число меток до/после дедупликации, например:
  ```
  extract: received text=128 chars, labels=3 (2 after dedup)
  extract_batch: received 5 texts, labels=3 (2 after dedup)
  ```
- **На каждый текст** — длина до и после очистки и на сколько чанков он разбит:
  ```
  Text 0: 128 chars -> 120 chars after cleaning, 1 chunk(s)
  ```
  Если после очистки текст оказался пустым — отдельная запись `Text 0: 5 chars -> empty after cleaning, skipped`.
- **Если текст разбит на несколько чанков** — дополнительно строка о разбиении и прогресс по каждому чанку:
  ```
  Text 0: 620 words (4200 chars) exceeds chunk size 300 — split into 3 chunks
  Text 0 chunk 1/3 (offset 0, 1400 chars): extracted 4 entities
  Text 0 chunk 2/3 (offset 1400, 1400 chars): extracted 2 entities
  Text 0 chunk 3/3 (offset 2800, 1400 chars): extracted 1 entities
  ```
- **Итог по тексту** — общее число найденных сущностей и разбивка по меткам:
  ```
  Text 0: 7 entities found from 3 chunk(s): {'person': 3, 'city': 4}
  ```

Ошибки инференса модели логируются с уровнем `ERROR` (с traceback) — см. строку "Ошибки модели" в разделе "Контракт".

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
