# 17. Парный transfer-vs-scratch curriculum

## Назначение

Новый контур проверяет один соседний переход `A→B`, `B→C`, `C→D` или
`D→E`. Он не меняет stage внутри существующего `TrainingRunner`: вместо этого
обучает два независимых target-stage run с одинаковым бюджетом и окружением.

- `transfer` получает только веса source-модели и новый target-stage optimizer;
- `scratch` получает независимо и детерминированно инициализированную модель;
- оба run сохраняют обычные full checkpoint с optimizer, league, RNG и progress;
- при принятии решения следующий segment начинает ровно с full checkpoint
  transfer-arm. Scratch остаётся доказательным контролем.

## Запуск

Сначала создать B→C starter config и отредактировать бюджеты, seed'ы,
оппонентов, CI и calibration-пороги:

```bash
.venv/bin/python -m poker.curriculum_cli \
  --write-default-config configs/b-to-c-paired.json
```

Затем указать два разных immutable full checkpoint source-stage. Reference —
закреплённый предыдущий checkpoint для отдельной проверки регрессии:

```bash
.venv/bin/python -m poker.curriculum_cli \
  --config configs/b-to-c-paired.json \
  --run-dir runs/b-to-c-paired \
  --source-checkpoint runs/hu-b/checkpoints/latest.pt \
  --reference-checkpoint runs/hu-b-reference/checkpoints/latest.pt
```

Первый `Ctrl+C` останавливает rung на целой границе PPO iteration и оставляет
resumable full checkpoint. Повтор той же команды проверяет intent/hash'и и
продолжает незавершённые arms. Подмена config, checkpoint или evaluation
artifact приводит к fail-closed остановке.

## Оппоненты evaluation

Каждый non-candidate slot задаётся явно и имеет стабильную identity. Runtime
принимает только allow-listed bot:

```json
{"identity": "rule", "provenance": {"kind": "bot", "bot": "rule"}}
```

или checkpoint с обязательным SHA-256:

```json
{
  "identity": "historical-c",
  "provenance": {
    "kind": "checkpoint",
    "path": "runs/archive/historical-c.pt",
    "sha256": "<64 hex>"
  }
}
```

Для каждого deal seed кандидат играет на каждом физическом месте, а
относительные opponent slots вращаются вместе с ним. Transfer и scratch
получают одинаковый deal/seat protocol; CI считается по целым common-deal
блокам, а не по отдельным коррелированным раздачам.

## Gate

Каждый target protocol (в том числе отдельные stack-варианты D→E) обязан
пройти независимо:

- нижняя граница paired CI `transfer − scratch`;
- нижняя граница target-stage BB/100 transfer;
- ноль/заданный максимум illegal actions;
- aggregate ECE/MAE `expected_showdown_share`;
- ECE, MAE и минимальный support каждого обязательного
  `street × active_players` stratum.

Отдельная source-stage suite сравнивает source candidate с reference на
source-правилах и гейтит нижнюю границу paired CI. Результаты разных stages не
сравниваются напрямую как один BB/100.

## Артефакты и границы утверждений

До mutable work атомарно записывается intent. Затем сохраняются native arm
checkpoints, frozen full artifacts, player-safe aggregate evaluation JSON,
decision report и manifest, связанные SHA-256. Принятое решение указывает на
full transfer checkpoint; отклонённое не публикует adoption path.

Это инфраструктура доказательного перехода, а не свидетельство, что уже
обучена сильная 3-/5-max стратегия. Для такого утверждения нужны реальные
run artifacts, достаточный support всех configured strata и принятый gate.
