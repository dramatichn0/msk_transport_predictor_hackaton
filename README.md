# Прогноз задержек наземного транспорта


## Дополнительная документация

- [Запуск проекта](docs/LAUNCH.md)
- [Подача потока и просмотр прогнозов](docs/STREAMING.md)
- [PyDoc/Sphinx и OpenAPI/Swagger](docs/API_AND_CODE_DOCUMENTATION.md)
- [Производительность и дополнительные возможности](docs/PERFORMANCE_AND_FEATURES.md)
- [Дашборд](docs/DASHBOARD.md)

## Current implementation status (September 26, 2026)

The current code includes: TCP NDTP listener on port 9201 with NPL/NPH/CRC/Nav00 parsing; decoded JSON door-cell features; schedule-constrained segment map matching; causal temporal CNN; CatBoost/LightGBM/sklearn ensemble; ONNX export/parity/optional quantization; What-if UI for a selected official `tr_id`; online absolute-error measurement after observed arrival; and pattern evidence in incident cards.

The supplied dataset still has no road-centerline network, no complete binary IRMA/Crown layouts, and no hidden validate labels. Therefore road-level matching, full raw-binary door decoding, hidden validate MAE, calibrated probabilities, TensorRT runtime, Docker build/run, and production p99 are not verified by this repository.

## Code review status

После review исправлены критичные проблемы первой версии:

- единый валидируемый контракт между backend и ML (`transit_common`); batch больше не теряет прогнозы из-за несовпадающих форматов;
- строгий горизонт `(T+10 минут, T+15 минут]`, UTC-нормализация, запрет будущих точек истории, `NaN/Infinity` и анонимных пакетов;
- chronological/purged holdout, отдельные regression и classification branches, калибровка риска больше не подменяется сигмоидой от регрессии;
- недоступный или необученный ML явно выдаёт `degraded=true`, `delay_probability=null`, `risk=unknown`;
- старые/дублирующиеся пакеты, зависшие состояния и ответы медленного инференса не перезаписывают более свежие данные;
- запросы batch ограничены 256 записями; история/состояние ограничены, replay сортирует CSV в памяти и отправляет его блоками через один HTTP connection pool;
- replay загружает расписание соответствующего split, а обучение не читает произвольные пути за пределами `DATASET_DIR`;
- Dashboard больше не вставляет телеметрию через `innerHTML`; what-if отключён вместо неподтверждённой оценки.

Проверка после review: `32` unit/integration теста проходят. Дополнительно:

```bash
python scripts/verify_review.py
```

Скрипт проводит реальный causal join train, временной purge, проверяет сохранение/загрузку модели и replay validate через ASGI HTTP. Отчёт записывается в `docs/review-verification.json`; это не замена проверке в Docker и нагрузочному тесту.

Демонстрационный, запускаемый стек раннего прогнозирования для московского датасета: API обработки потока и диспетчерский экран отделены от ML API и модели. Расписание является единственным источником остановок и геометрии; телеметрия не создаёт маршруты или остановки.

## Архитектура

```text
NDTP/эмулятор → внешний адаптер или CSV replay ──> Backend (FastAPI)
                                                     ├─ map matching к stop geom из schedule
                                                     ├─ состояние и инциденты для dashboard
                                                     └─ POST /predict/batch ──> ML service (FastAPI)
                                                                         ├─ CatBoost + sklearn tabular
                                                                         └─ PyTorch MLP по агрегатам истории
```

Backend и ML — независимые Docker-образы; их API со схемами OpenAPI доступны на `/docs` и `:8001/docs`. Эмулятор в `dataset/` документирован как TCP NDTP-клиент. Для него нужен совместимый NDTP-приёмник/адаптер, преобразующий пакеты G6CellNav00 в JSON схему `/api/telemetry`; приложение также принимает CSV replay. Не следует направлять эмулятор прямо на HTTP-порт FastAPI.

## Быстрый запуск

Требуется Docker с Compose:

```bash
docker compose up --build
```

- Дашборд и backend: http://localhost:8000
- Backend OpenAPI: http://localhost:8000/docs
- ML OpenAPI: http://localhost:8001/docs
- Проверка состояния: `GET /health`, `GET /api/state`

На старте backend загружает `dataset/validate/schedule_plan.csv` (fallback: `train/schedule.csv`). Модель обучается отдельно:

```bash
curl -X POST http://localhost:8001/train/dataset
```

Веса берутся из каталога `./models`, примонтированного в ML-контейнер как
`/app/models`. Без обученных весов сервис работает с прозрачным baseline
`cur_dev_s`, а при недоступности ML backend деградирует к известному отклонению.
Датасет в backend контейнере доступен только для чтения.

## Данные и горизонт

Исходные файлы находятся в `dataset/`: `train|test/traffic.csv`, расписания, `labels/labels_train.csv`, `labels_test.csv`, а также `validate/traffic.csv`, `schedule_plan.csv`, `points.csv` и `sample_submission.csv`. CSV кодированы UTF-8, разделитель запятая, кроме submission, где разделитель `;`.

Цель соответствует контракту набора: задержка в секундах на целевой остановке относительно планового времени, с использованием только телеметрии на момент `T` и `cur_dev_s`. Для прогноза API требует плановое время остановки строго в диапазоне `(T+10 минут, T+15 минут]`; вне окна вернётся HTTP 422. Validate не содержит фактических задержек. Неизвестные `tr_id` сохраняются как контекстные ТС, но им не строится прогноз.

`POST /api/telemetry` принимает JSON поля `tr_id`, `unit_id`, обязательный `event_time`, `lat`, `lon`, `speed` (км/ч), `heading` (градусы), `location_valid`, опциональный `cur_dev_s`. Нужен хотя бы один идентификатор. `unit_id` разрешается через однозначное соответствие из референсного CSV; он никогда не приравнивается к `tr_id`. Raw NDTP не декодируется этим endpoint.

Времена без offset в датасете интерпретируются как UTC; времена с offset преобразуются в UTC. Это явное соглашение требует подтверждения для будущих внешних источников. Невалидные координаты становятся `null`, скорость вне диапазона 0–180 км/ч отбрасывается. `/api/telemetry/batch` принимает до 256 пакетов, ошибки отдельных записей изолируются. Старые и повторные события пропускаются. `/api/replay/{train|test|validate}` загружает соответствующее расписание, очищает состояние и сортирует CSV; во время replay live-ingestion отвечает 409.

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Статус, число ТС/остановок и метрики обработки |
| GET | `/api/schedule` | Официальные остановки/координаты по `tr_id` |
| POST | `/api/telemetry` | Приём одной телеметрической записи и прогноз при доступной цели в горизонте |
| POST | `/api/telemetry/batch` | Пакетная обработка с изоляцией ошибок |
| POST | `/api/replay/{split}` | Replay исторического CSV |
| GET | `/api/state` | Состояние ТС и инцидентов диспетчерского экрана |
| POST | `/api/what-if` | ????????? ?????? ??????? ??? ???????????? `tr_id` ? ?????? ??????????? |
| GET | `/health` | ML health; `degraded` до обучения (порт 8001) |
| POST | `/predict`, `/predict/batch` | Строгий контракт ML, batch: `{"predictions": [...]}` |
| POST | `/train`, `/train/dataset` | Переобучение и атомарная публикация модели |

FastAPI генерирует детальную OpenAPI схему, включая поля запросов и ответы.

## Обучение и ограничения интерпретации

Обучение использует `target_delay_s`, последние 20% точек по времени — holdout. Из train удаляются точки, чьи плановые/фактические целевые прибытия ещё не известны на начало holdout. MAE считается по тому же ансамблю, который обслуживает запросы; также выводятся baseline MAE, AUC и Brier. Поле `predicted_absolute_error_seconds` удалено: абсолютная ошибка требует фактического target, неизвестного в момент прогноза.

Вероятность берётся из отдельного классификатора, а не сигмоиды регрессионного отклонения. Она **не объявляется калиброванной**. CatBoost и PyTorch включаются при наличии зависимостей. PyTorch-ветка пока MLP по табличным агрегатам, **не sequence/Transformer модель**. Контекст и расстояние вычисляются backend, но пока не включаются в модель, чтобы не подавать признаки, отсутствующие при обучении.

Обучение и инференс выполняются вне ASGI event loop; concurrent training jobs отклоняются с 409. Старый snapshot обслуживает запросы до атомарной публикации новой версии. Артефакты первой версии несовместимы — требуется переобучение. `joblib` разрешено загружать только из доверенного model volume. Локальная проверка: Python 3.11; Docker: Python 3.12.

Map matching — ближайшая остановка в радиусе 500 м с временным разрешением повторных посещений. Это **не HMM** и не геометрия улично-дорожной сети. Отклонение оценивается при обнаруженном прибытии (≤60 м, ≤3 км/ч) либо передаётся как `cur_dev_s`; неизвестное значение не выдаётся за гарантированный ноль.

### Оставшиеся production-блокеры

- Нет бинарного NDTP TCP adapter/handshake.
- Нет настоящей sequence-модели, подтверждённой калибровки вероятностей, SLA <1 с или доказанного превосходства над persistence baseline.
- Backend хранит состояние в памяти; нужен один worker. Для горизонтального масштабирования — внешнее хранилище/партиционирование.
- CSV replay сортируется целиком в памяти: для больших исторических архивов нужен внешний merge/index.
- TLS/auth/RBAC/rate limits должны быть реализованы на ingress. Compose публикует порты только на `127.0.0.1`; не открывайте сервисы напрямую в интернет.
- MapLibre GL JS, Google Fonts и растровые тайлы OpenStreetMap требуют внешней сети.
  Для крупных публичных развёртываний настройте собственный совместимый tile provider;
  визуальный браузерный тест не проводился.

Эти ограничения означают, что стек остаётся прототипом, а не полностью принятым production-решением.

## Проверки

```bash
python -m unittest discover -s tests -v
```
