# Рабочая система из трёх модулей

**Проект:** [GitHub-репозиторий msk_transport_predictor_hackaton](https://github.com/dramatichn0/msk_transport_predictor_hackaton)

Главный экран дашборда содержит карту маршрутов, текущие транспортные средства и сводные показатели.

![Главный экран дашборда: карта маршрутов, текущие транспортные средства и сводные показатели](screenshots/main_white.png)


README в корне репозитория содержит сценарий запуска. Система представляет
собой продукт для прогнозирования изменений в графике
наземного транспорта.

## Что получает пользователь

Система принимает телеметрию, связывает её с официальным расписанием,
оценивает отклонение на горизонте 10–15 минут и показывает результат
диспетчеру на карте. В едином контуре доступны:

- прогноз задержки и отдельная оценка риска;
- карта маршрутов и позиций транспортных средств;
- карточки транспорта и история инцидентов;
- потоковая обработка NDTP и исторический replay;
- сводные показатели и online MAE;
- Swagger/OpenAPI для интеграции с внешними системами.

## Архитектура продукта

| Модуль | Роль | Технологии |
|---|---|---|
| ML-ядро | Подготовка признаков, обучение и инференс | Python, FastAPI, NumPy, scikit-learn; доступны ветки LightGBM, CatBoost, PyTorch и ONNX Runtime |
| Backend | Приём телеметрии, map matching, состояние, инциденты и API | Python, FastAPI, Pydantic, HTTPX, asyncio |
| BI-дашборд | Карта, показатели, транспорт и инциденты | HTML, CSS, JavaScript, MapLibre GL JS, GeoJSON |

Общие модели `transit_common` задают единый контракт Backend ↔ ML.
Все компоненты запускаются через Docker Compose.

## Запуск

Из корня репозитория:

```powershell
docker compose up -d --build
```

| Компонент | Адрес |
|---|---|
| Дашборд и Backend | `http://localhost:8000/` |
| Backend Swagger | `http://localhost:8000/docs` |
| ML Swagger | `http://localhost:8001/docs` |
| NDTP TCP listener | `localhost:9201` |

Проверка:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8001/health
```

Остановка:

```powershell
docker compose down
```