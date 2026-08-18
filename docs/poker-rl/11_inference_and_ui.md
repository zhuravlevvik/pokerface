# 11. Инференс-сервис и наблюдаемый стол

## Результат

Пользователь наблюдает раздачу в реальном времени, а обучение изолировано от пользовательского интерфейса.

## Компоненты

```text
Trainer -> model registry/checkpoints
Inference service -> загружает фиксированный checkpoint
Game server -> проводит одну наблюдаемую раздачу
Web client -> показывает стол и объяснения
```

Game server общается с клиентом через WebSocket: это позволяет показывать каждое действие, анимации и обновление вероятностей без перезагрузки страницы.

## Ответ модели

```json
{
  "action_probabilities": {"fold": 0.12, "call": 0.38, "raise": 0.50},
  "bet_size_probabilities": {"half_pot": 0.46, "pot": 0.31, "all_in": 0.05},
  "equity": {"win": 0.61, "tie": 0.03, "loss": 0.36, "total": 0.625},
  "value_bb": 3.7
}
```

## UI

- стол, позиции, стеки, банк, карты и текущий ход;
- история действий и выбранный рейз-сайзинг;
- текущая equity и график её изменения: preflop → flop → turn → river;
- распределение действий и value в BB;
- режим игрока (без чужих карт) и режим наблюдателя (раскрытие после окончания раздачи);
- импорт и воспроизведение replay.

## Критерий приёмки

Одна сохранённая раздача точно воспроизводится в браузере; отображаемые данные соответствуют состоянию и ответу inference-сервиса.

## Реализация и запуск

Реализация разделена на три слоя:

```text
poker.inference     fixed checkpoint -> InferenceResponse
poker.game_server   state machine + serialisable observer/replay events
poker.web           optional FastAPI HTTP/WebSocket adapter + static browser UI
```

Для запуска UI установите optional web-зависимости и передайте ASGI-приложение
в Uvicorn:

```bash
.venv/bin/pip install -e '.[web]'
.venv/bin/uvicorn poker.web:create_app --factory --reload
```

После запуска откройте `http://127.0.0.1:8000`. Браузер посылает в
`/ws/table` команды `start_hand` или `replay` и получает отдельные сообщения
`hand_started`, `action`, `hand_complete`. REST-вариант для интеграций —
`POST /api/hand`, health check — `GET /api/health`.

По умолчанию используется `HeuristicInferenceService`, чтобы UI был пригоден
сразу после checkout. Это не обученная модель и его equity — только proxy
силы руки. Для реальной модели создайте `GameServer` с
`CheckpointInferenceService.from_checkpoint(path)` и передайте его в
`create_app`; сервис читает checkpoint, но никогда его не меняет.

В player-mode чужие карты не отправляются ни во время, ни после раздачи.
Spectator-mode раскрывает все карты только в terminal snapshot. Оба replay
содержат seed и action log и потому детерминированно импортируются обратно;
player replay при этом не требует раскрывать карты соперников.

## Mixed-policy просмотр

Стол поддерживает 2, 3 и 5 игроков. Для каждого места в UI можно выбрать
baseline-бота или checkpoint из каталога, настроенного на сервере. Браузер
передаёт только безопасный id вида `bot:rule` или `checkpoint:hu-best`; путь
к файлу checkpoint остаётся на стороне процесса сервера.

Удобный запуск каталога и заранее выбранных мест:

```bash
.venv/bin/python -m poker.watch \
  --checkpoint hu-best=runs/hu/checkpoints/best.pt \
  --seat checkpoint:hu-best --seat bot:rule \
  --players 2 --hands 3 --seed 42000
```

При editable install ту же команду можно запустить как `pokerface-watch`.

В интерфейсе доступны серия раздач, пауза, покадровый переход по ходам,
переход к следующей раздаче, скорость воспроизведения и экспорт последнего
replay. Серия использует последовательные seed (`seed`, `seed + 1`, ...), а
кнопка download сохраняет самодостаточный replay выбранной раздачи.
