# Документация кода и API

## PyDoc и docstrings

Ключевые Python-модули содержат docstrings, совместимые с PyDoc:

- `backend/app/main.py` — NDTP, поток, map matching и Backend API;
- `ml_service/app/main.py` — feature pipeline, обучение и инференс;
- `transit_common/contracts.py` — общий контракт Backend ↔ ML;
- `scripts/create_submission.py` — обучение/загрузка модели и генерация CSV.

Посмотреть документацию модуля:

```powershell
python -m pydoc backend.app.main
python -m pydoc ml_service.app.main
python -m pydoc transit_common.contracts
```

Открыть PyDoc в браузере:

```powershell
python -m pydoc -b
```

После запуска команда выведет локальный адрес HTTP-сервера PyDoc.

## Быстрая проверка документации и типов

```powershell
python -m compileall -q backend ml_service transit_common scripts
```

Тесты:

```powershell
python -m pytest -q tests/test_system.py
```

JavaScript дашборда:

```powershell
node --check dashboard\app.js
```

## OpenAPI / Swagger

После запуска Docker:

- Backend Swagger UI: <http://localhost:8000/docs>
- Backend OpenAPI JSON: <http://localhost:8000/openapi.json>
- Backend ReDoc: <http://localhost:8000/redoc>
- ML Swagger UI: <http://localhost:8001/docs>
- ML OpenAPI JSON: <http://localhost:8001/openapi.json>
- ML ReDoc: <http://localhost:8001/redoc>

Сохранить спецификации в файлы:

```powershell
Invoke-WebRequest http://localhost:8000/openapi.json `
  -OutFile docs\backend-openapi.json

Invoke-WebRequest http://localhost:8001/openapi.json `
  -OutFile docs\ml-openapi.json
```

## Backend API

| Метод | Endpoint | Назначение |
|---|---|---|
| `GET` | `/health` | Состояние расписания, потока, TCP NDTP и online MAE |
| `GET` | `/api/schedule` | Официальные остановки и координаты по `tr_id` |
| `POST` | `/api/telemetry` | Один нормализованный телематический пакет |
| `POST` | `/api/telemetry/batch` | До 256 пакетов с изоляцией ошибок |
| `POST` | `/api/telemetry/ndtp/nav` | JSON-представление `G6CellNav00` |
| `POST` | `/api/telemetry/ndtp/nav-cell/{unit_id}` | Raw 26-byte Nav00 cell |
| `POST` | `/api/replay/{train\|test\|validate}` | Исторический CSV replay |
| `GET` | `/api/state` | Текущие ТС, инциденты и метрики |
| `POST` | `/api/what-if` | Гипотетическая оценка выпуска дополнительного ТС |

## ML API

| Метод | Endpoint | Назначение |
|---|---|---|
| `GET` | `/health` | Активная модель, provider и метрики обучения |
| `POST` | `/predict` | Один прогноз |
| `POST` | `/predict/batch` | Пакет прогнозов |
| `POST` | `/train/dataset` | Обучение на штатном causal join |
| `POST` | `/train` | Обучение по разрешённому labels CSV |

## Основные ограничения контракта

- горизонт строго `(T+10 минут, T+15 минут]`;
- история содержит только точки с `event_time <= T`;
- `NaN` и `Infinity` запрещены;
- batch ограничен 256 записями;
- прогноз строится только для ТС из официального расписания;
- неизвестные ТС не вызывают ошибку;
- модель возвращает `degraded=true`, если используется fallback.

## Пример запроса ML

```powershell
$request = @{
  tr_id = "115106"
  event_time = "2026-01-06T12:00:00Z"
  target_time_begin = "2026-01-06T12:12:00Z"
  target_stop_id = "official-stop-id"
  cur_dev_s = 60
  speed = 24
  lat = 55.7551234
  lon = 37.617321
  heading = 90
  history = @()
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post `
  -Uri http://localhost:8001/predict `
  -ContentType "application/json" `
  -Body $request
```
