# Подача телематического потока и просмотр прогнозов

Backend принимает два режима данных:

1. исторический CSV replay;
2. realtime NDTP-поток от эмулятора по TCP.

Расписание является единственным источником официальных остановок и ниток.
ТС, которых нет в расписании, не вызывают ошибку и используются только как
контекстные машины.

## Вариант 1. Исторический датасет

Запустить систему:

```powershell
docker compose up -d --build
```

Replay тестового или валидационного потока:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/replay/test
```

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/replay/validate
```

Для replay используются:

- `dataset/test/traffic.csv` + `dataset/test/schedule.csv`;
- `dataset/validate/traffic.csv` + `dataset/validate/schedule_plan.csv`.

Replay сортирует исторические пакеты по времени и подаёт их в тот же backend
pipeline, который используется для realtime-потока.

Во время replay новые live-запросы телеметрии временно получают HTTP `409`.

Проверить итог:

```powershell
Invoke-RestMethod http://localhost:8000/api/state
```

Поля результата:

- `vehicles` — последние состояния ТС;
- `incidents` — история алертов;
- `metrics` — количество пакетов, неизвестных ТС, fallback ML и измеренных
  ошибок.

## Вариант 2. NDTP-эмулятор

В проекте находится архив эмулятора:

```text
dataset/ndtp-telemetry-emulator.tar
```

Загрузить образ:

```powershell
docker load -i dataset\ndtp-telemetry-emulator.tar
```

Запустить эмулятор:

```powershell
docker run --rm -p 18080:18080 `
  --add-host=host.docker.internal:host-gateway `
  --name ndtp-emu `
  ndtp-telemetry-emulator:1.0
```

Backend слушает NDTP TCP на:

```text
host: 127.0.0.1
port: 9201
```

Для эмулятора, запущенного в Docker, укажи:

```json
{
  "targetHost": "host.docker.internal",
  "targetPort": 9201,
  "units": [
    {
      "unitId": 1166336,
      "intervalMs": 5000,
      "autoGenerate": true,
      "cells": []
    }
  ]
}
```

Сохрани JSON, например, в `emulator-config.json`, и отправь:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:18080/api/config `
  -ContentType "application/json" `
  -InFile .\emulator-config.json
```

Проверить API эмулятора:

```powershell
Invoke-RestMethod http://localhost:18080/api/cells
Invoke-RestMethod http://localhost:18080/api/config
```

Остановка эмуляции:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://localhost:18080/api/config `
  -ContentType "application/json" `
  -Body '{"targetHost":"host.docker.internal","targetPort":9201,"units":[]}'
```

Эмулятор отправляет NDTP-пакеты `G6CellNav00`. Backend декодирует координаты,
скорость, курс и валидность GPS, затем выполняет map matching к официальному
расписанию.

## Прямой JSON-запрос в Backend

Для ручной проверки можно отправить уже нормализованный пакет:

```powershell
$packet = @{
  tr_id = "115106"
  unit_id = "664030"
  event_time = "2026-01-06T12:00:00Z"
  lat = 55.7551234
  lon = 37.617321
  speed = 24
  heading = 90
  location_valid = $true
  cur_dev_s = 60
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://localhost:8000/api/telemetry `
  -ContentType "application/json" `
  -Body $packet
```

## Где смотреть результат

Открыть дашборд:

```text
http://localhost:8000/
```

На карте отображаются:

- официальные нитки расписания;
- scheduled ТС;
- контекстные ТС синим цветом;
- зелёные, жёлтые, красные и серые состояния риска;
- красные/жёлтые/зелёные/серые точки инцидентов.

История инцидентов находится справа. Нажатие на инцидент центрирует карту на
его последней известной позиции.

## Метрики и алерты через API

```powershell
Invoke-RestMethod http://localhost:8000/api/state
```

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Полезные поля `metrics`:

- `packets` — принятые пакеты;
- `invalid` — некорректные пакеты;
- `unknown_vehicles` — ТС без расписания;
- `predictions` — выполненные прогнозы;
- `ml_fallbacks` — случаи деградации к известному отклонению;
- `incidents` — количество созданных инцидентов;
- `measured_error_count` — измеренные прогнозы после фактического прибытия;
- `measured_absolute_error_seconds` — сумма абсолютных ошибок.
