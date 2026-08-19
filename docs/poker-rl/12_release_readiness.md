# Release-readiness audit

Дата аудита: 2026-08-19.

Этот документ фиксирует состояние **текущего checkout**, а не планируемое
состояние проекта. «Реализовано» ниже означает, что в репозитории есть код и
автоматическая проверка; это не означает, что уже получена сильная покерная
стратегия. В частности, в Git нет сохранённого обученного checkpoint'а,
evaluation-отчёта такого checkpoint'а или результатов длительного self-play.

## Краткий вывод

Игровая, обучающая и демонстрационная инфраструктура реализована и покрыта
автоматическими тестами. Полный тестовый набор проходит. Готового агента с
доказанной силой пока нет, поэтому проект нельзя позиционировать как
обученный 5-max покерный бот и нельзя считать пройденными M4--M6.

## Реально выполненные проверки

В этом checkout выполнены следующие команды:

```bash
.venv/bin/python -m pytest
# 218 passed, 2 warnings

.venv/bin/python -m compileall -q poker
# успех

.venv/bin/python -c "from poker.game_server import GameServer; events=GameServer().observe_hand(seed=20260812, mode='spectator'); assert events[-1]['type']=='hand_complete'; assert events[-1]['replay']['seed']==20260812; print(f'events={len(events)} actions={len(events[-1][\"replay\"][\"actions\"])}')"
# events=19 actions=17

.venv/bin/python -c "from poker.web import create_app; app=create_app(); assert app.title == 'Pokerface observer'; print(app.title, app.version)"
# Pokerface observer 0.1.0
```

Дополнительно evaluation API был выполнен на пяти фиксированных раздачах
между `TightBot` и `RuleBot`; он создал JSON в
`/private/tmp/pokerface-audit-evaluation.json`. Это smoke-check формата
отчёта, а не измерение качества модели: candidate не является нейросетью,
поэтому equity/model diagnostics в отчёте отсутствуют, а пяти раздач
недостаточно для статистического вывода.

Наблюдаемые предупреждения pytest относятся к окружению (`numpy` не
установлен для PyTorch) и prototype nested tensors в PyTorch; ни один тест не
завершился ошибкой. `numpy` не является зависимостью проекта и не требуется
для пройденных проверок.

## Статус вех

| Веха | Статус | Доказательства в коде | Чего не хватает для закрытия |
| --- | --- | --- | --- |
| M1 — игровая основа | **Пройдена для кода** | Детерминированный NLHE engine: `poker/game_state.py`, `poker/betting.py`, `poker/evaluator.py`, replay/traces; сценарии в `tests/test_hand_state.py`, `tests/test_cards_evaluator.py`, `tests/test_rules.py`, `tests/test_simulator_and_traces.py`. | Ничего из формулировки M1. Поддержка 5-max, side pots и exact replay уже тестируется. |
| M2 — наблюдаемый покер без RL | **Частично** | Baseline-боты и турниры: `poker/bots.py`, `poker/tournament.py`; веб-стол и replay: `poker/game_server.py`, `poker/web.py`, `poker/static/`; тесты `test_baseline_bots.py`, `test_game_server.py`. | Нет режима, в котором человек сам выбирает ход за hero seat: текущий `GameServer` запрашивает `DecisionService` для каждого места. Нет browser E2E-теста реального рендеринга. |
| M3 — состояние и equity | **Инфраструктура готова; приёмка не доказана** | Versioned observation: `poker/observation.py`; модель с policy, bet-size, value и equity: `poker/model.py`; virtual-showdown labels и calibration metrics: `poker/equity.py`, `poker/traces.py`; restricted reproducibility corpus, resumable pretrainer и holdout report: `poker/pretraining_data.py`, `poker/pretraining.py`, `poker/pretraining_runner.py`; тесты `test_observation_environment.py`, `test_model.py`, `test_equity_labels.py`, `test_pretraining*.py`. | Нет сохранённого отчёта калибровки реально предобученной модели. Поэтому нельзя утверждать, что equity калибрована, лишь что подсчёт label, обучение и формат отчёта реализованы. |
| M4 — heads-up self-play | **Не пройдена; инфраструктура gate готова** | PPO/GAE и batched policy sampling имеются в `poker/training.py`; heads-up stage описан в `poker/curriculum.py`; `poker.train_cli` запускает воспроизводимый resumable run с graceful `Ctrl+C` и warm-start из pretraining checkpoint; `poker/promotion.py` выполняет paired-seed HU evaluation, immutable candidate/archive и CI/calibration/regression gates. | Нет реально обученного и promoted heads-up checkpoint'а с сохранённым evaluation report. Нельзя заявлять превосходство над baseline. |
| M5 — 3-max и 5-max | **Не пройдена; инфраструктура gate готова** | Curriculum A--E и expected-showdown-share head: `poker/curriculum.py`, `poker/equity.py`; matched target-stage transfer/scratch full runs: `poker/paired_rung.py`; heterogeneous common-deal seat rotation и paired CI: `poker/multiway_evaluation.py`; adjacent-stage durable coordinator и CLI: `poker/curriculum_coordinator.py`, `poker.curriculum_cli`; end-to-end proof: `test_curriculum_end_to_end.py`. | Нет принятых report/checkpoint artifacts длительных B→C→D→E runs, archive реальных стратегий или подтверждённой устойчивости 3-/5-max модели. |
| M6 — эксплуатационная готовность | **Частично** | Versioned model/full-run checkpoints: `poker/model.py`, `poker/train_runner.py`; atomic save, optimizer/RNG/league resume и finite-health gate; hash-проверяемые HU/curriculum evidence; immutable per-iteration experiment ledger и deterministic tuning comparison: `poker/experiments.py`, `poker/experiment_runner.py`, `poker/tuning.py`; pinned opponent registry and player-safe aggregate reports. | Нет опубликованного evaluation report реального обученного checkpoint'а и versioned HTTP API (`/api/v1/...`). Наличие registry/ledger-кода не заменяет реальный release artifact. |
| M7 — демонстрационный продукт | **Частично** | HTTP/WebSocket UI, action history, equity graph, player/spectator modes и replay реализованы в `poker/web.py`, `poker/game_server.py`, `poker/static/`; mixed-policy UI и `poker.watch` запускают 2/3/5-max checkpoint-vs-bot/checkpoint серии. Тесты проверяют safe checkpoint catalog и отсутствие утечки opponent analysis в player-mode. | Нет browser E2E-подтверждения и демонстрационного replay от реально обученного checkpoint'а, привязанного к release. |

## Связь с коммитами

| Коммит | Реализованный слой |
| --- | --- |
| `c6f1307` | детерминированный NLHE engine |
| `14a35c8` | batched simulator и training traces |
| `15d2e94` | baseline bots и tournaments |
| `1cc5991` | versioned observations |
| `15ccc2e` | multi-head agent model |
| `a5def4e` | equity supervision и calibration metrics |
| `88f644e` | staged curriculum и checkpoint transfer |
| `78e0025` | PPO league self-play training infrastructure |
| `9e64b30` | reproducible evaluation suite |
| `c09f031` | observable inference web interface |

## Воспроизводимые команды для следующего release-candidate

Базовая валидация:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m compileall -q poker
```

Сформировать machine-readable holdout report для **реального** checkpoint'а
следует через `poker.evaluation.evaluate_suite(...)`, передав
`ModelPolicy.from_checkpoint(...)`, фиксированный `EvaluationConfig` и
baseline/historical opponents. Сохранённый JSON должен храниться рядом с
checkpoint'ом и содержать commit SHA, config, seed range и хеш checkpoint'а;
сейчас такого release-артефакта нет.

Запустить UI (после установки web extra):

```bash
.venv/bin/pip install -e '.[web]'
.venv/bin/uvicorn poker.web:create_app --factory --reload
```

Он доступен по адресу `http://127.0.0.1:8000`; по умолчанию это
детерминированный эвристический демонстратор, а не trained-agent release.

## Объективные критерии следующего перехода

1. Выполнить длительное обучение stage A (heads-up) с зафиксированной
   конфигурацией, seed'ами и бюджетом шагов; сохранить checkpoint.
2. Сгенерировать holdout JSON на существенно большем фиксированном числе рук
   против каждого baseline и historical opponent. Заранее записать порог
   `BB/100`, доверительный интервал/повторные seed'ы и максимальный ECE.
3. Закрыть M4 только если checkpoint превосходит baseline по этому протоколу
   и его equity-метрики соответствуют порогу `CurriculumConfig`.
4. Для каждого следующего curriculum stage запустить `poker.curriculum_cli`,
   сохранить coordinator intent/report/manifest и принятый full transfer
   checkpoint. Наличие реализации paired rung не заменяет реальные результаты.
5. До заявки M6 зарегистрировать checkpoint и evaluation в release registry,
   затем добавить versioned HTTP API.
6. До заявки M7 добавить ручной режим hero (если он остаётся частью
   требований), browser E2E и прикреплённый demonstration replay от
   зафиксированного checkpoint'а.
