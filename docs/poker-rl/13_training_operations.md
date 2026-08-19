# 13. Запуск, остановка и продолжение обучения

## Первый запуск

Установить RL-зависимости и создать стартовый JSON-конфиг:

```bash
.venv/bin/pip install -e '.[rl]'
.venv/bin/python -m poker.train_cli \
  --write-default-config configs/hu-stage-a.json
```

Запустить отдельный run directory:

```bash
.venv/bin/python -m poker.train_cli \
  --config configs/hu-stage-a.json \
  --run-dir runs/hu-stage-a
```

Конфиг фиксирует model, PPO, curriculum, league, seed, число рук на PPO
iteration и частоту checkpoint'ов. Stage A использует heads-up, 100 BB и
сокращённый набор сайзингов из `poker.curriculum`.

## Безопасная остановка

Первый `Ctrl+C` запрашивает graceful shutdown. Тренер:

1. завершает активный batch раздач;
2. выполняет один целостный PPO update по собранному rollout;
3. атомарно сохраняет `checkpoints/interrupt_<iteration>.pt` и `latest.pt`;
4. завершает процесс.

Второй `Ctrl+C` прерывает процесс немедленно и намеренно не сохраняет
потенциально частично обновлённое состояние. Периодические checkpoint'ы
защищают прогресс при аварийном завершении, когда обработать сигнал невозможно.

Checkpoint содержит веса модели, optimizer, stage, counters рук/решений,
состояние league, Python/PyTorch/CUDA RNG, seed следующей раздачи, конфиг и
manifest. Запись fsync'ится во временный файл и публикуется atomic rename.

## Продолжение

Продолжить последний run до target iteration из исходного конфига:

```bash
.venv/bin/python -m poker.train_cli \
  --run-dir runs/hu-stage-a \
  --resume latest
```

Либо указать конкретный checkpoint и новый абсолютный target iteration:

```bash
.venv/bin/python -m poker.train_cli \
  --resume runs/hu-stage-a/checkpoints/interrupt_00000025.pt \
  --iterations 100
```

Resume восстанавливает генераторы случайных чисел после создания модели и
league, поэтому следующий rollout продолжает исходный детерминированный поток.

## Новый PPO-run из warm-start checkpoint

Equity/pretraining checkpoint (или обычный model checkpoint) можно использовать
только как начальные веса для **нового** PPO-run:

```bash
.venv/bin/python -m poker.train_cli \
  --config configs/hu-stage-a.json \
  --run-dir runs/hu-stage-a-warm \
  --init-checkpoint runs/pretrain-stage-a/checkpoints/latest.pt
```

Альтернатива — задать `"init_checkpoint": "..."` на верхнем уровне run
config. Архитектура и versioned model metadata обязаны точно совпадать с
`model` в новом config. Инициализация переносит исключительно model weights:
optimizer, RNG, league, счётчики и manifest создаются заново. `--resume` и
`--init-checkpoint` взаимоисключающие; для продолжения полного PPO-run нужен
только `--resume`.

Если включён блок `promotion`, runner после заданного числа завершённых PPO
iterations сначала публикует frozen candidate, затем выполняет фиксированный
HU evaluation. Evaluation не меняет RNG обучаемого процесса; принятое решение
и состав исторической лиги входят в следующий full-run checkpoint.

## Opt-in переход Stage A → B

Вместо promotion Stage A run может включить блок `transition`; два механизма
взаимоисключающие. Автоматизация требует Stage A full-run
`transition.reference_checkpoint` и закреплённый
`transition.reference_checkpoint_sha256`, запускается только из A в B и требует
`curriculum.require_transfer_beats_scratch=false`, поскольку paired scratch
rung пока не встроен. Пример полного конфига и трактовка этого ограничения —
в [16_curriculum_transition.md](16_curriculum_transition.md).

На принятом переходе runner замораживает source, публикует evidence/report и
transfer artifact, переводит stage на B, сбрасывает optimizer по умолчанию и
ставит LR `base_learning_rate × 0.7`. Указание `reset_optimizer=false`
сохраняет optimizer moments, но не отменяет Stage B LR scale.

Если `Ctrl+C` пришёл на transition boundary, сохранённый pending state
дорабатывается при `--resume` до следующего rollout. Не удаляйте вручную
`curriculum-transitions/`: resume проверяет hash manifest, candidate, report
и transfer artifact и остановится при несогласованном состоянии.

## Парный переход между любыми соседними stages

Для настоящего transfer-vs-scratch сравнения используется отдельная команда,
которая сохраняет `TrainingRunner` single-stage и создаёт два target-stage
segment:

```bash
.venv/bin/python -m poker.curriculum_cli \
  --write-default-config configs/b-to-c-paired.json

.venv/bin/python -m poker.curriculum_cli \
  --config configs/b-to-c-paired.json \
  --run-dir runs/b-to-c-paired \
  --source-checkpoint runs/hu-b/checkpoints/latest.pt \
  --reference-checkpoint runs/hu-b-reference/checkpoints/latest.pt
```

Команда допускает только соседнюю пару, fixed bot или SHA-pinned checkpoint
для каждого evaluation seat, common-deal multiway CI и обязательные
expected-showdown-share strata. Первый `Ctrl+C` сохраняет целую границу arm;
та же команда продолжает по intent/manifest. Подробный контракт — в
[17_paired_multiway_curriculum.md](17_paired_multiway_curriculum.md).

## Просмотр checkpoint'а

Full training checkpoint совместим с inference loader. Запустить несколько
наблюдаемых раздач checkpoint против бота можно так:

```bash
.venv/bin/python -m poker.watch \
  --checkpoint hu-best=runs/hu-stage-a/checkpoints/latest.pt \
  --seat checkpoint:hu-best --seat bot:rule \
  --players 2 --hands 3 --seed 42000
```

После запуска открыть `http://127.0.0.1:8000`. Пути checkpoint'ов задаются
только в operator-controlled CLI; браузер работает с безопасными catalog id.
