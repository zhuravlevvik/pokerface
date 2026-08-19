# Experiment ledger, tuning и release evidence

Этот слой нужен до длительных запусков. Он не меняет PPO на ходу и не
объявляет модель сильной: он делает каждый trial воспроизводимым, сохраняет
полную кривую обучения и разрешает сравнивать только сопоставимые результаты.

## Один immutable trial

`pokerface-experiment` принимает отдельный `experiment.json`. В нём закреплены:

- полный `TrainingRunConfig` и точный budget iterations;
- SHA-256 фиксированного evaluation protocol;
- явный immutable `code_revision`;
- checkpoint на каждой завершённой PPO iteration.

Создать стартовый файл:

```bash
.venv/bin/python -m poker.experiment_cli \
  --write-default-config configs/experiment-a.json \
  --code-revision <commit-or-build-id> \
  --protocol-artifact configs/hu-fixed-evaluation.json
```

CLI сам записывает и хеширует preregistered HU protocol artifact; placeholder
revision не принимается. Затем:

```bash
.venv/bin/python -m poker.experiment_cli \
  --config configs/experiment-a.json \
  --run-dir runs/experiments/a-seed-11
```

Повтор той же команды продолжает `training/checkpoints/latest.pt`. Первый
`Ctrl+C` по-прежнему ждёт целую PPO boundary и публикует checkpoint; ledger
атомарно добавляет ровно одну запись на iteration. Если процесс упал после
checkpoint, но до ledger event, следующий запуск восстанавливает хвост из
immutable checkpoint record native manifest.

Источник истины — hash-chain в `experiment-ledger/events/`. Файл
`metrics.jsonl` является проверяемой проекцией и пересобирается из цепочки.
Записи содержат только aggregate counters/losses/KL/clip/entropy/gradient norm
и SHA checkpoint'а: наблюдения, карты и rollout туда не попадают.

## Fail-closed training health

Перед optimizer step/publish проверяются loss-компоненты, gradient norm,
параметры модели, Adam state и итоговые метрики. При NaN/Inf:

- iteration/counters и `latest.pt` не продвигаются;
- текущий in-memory runner становится непубликуемым;
- experiment пишет ограниченный `failure.json` и статус `failed_nonfinite`;
- продолжение возможно только из последнего durable checkpoint.

## Детерминированный sweep

Создать и отредактировать sweep:

```bash
.venv/bin/python -m poker.tuning_cli \
  --write-default-config configs/ppo-sweep.json \
  --code-revision <commit-or-build-id> \
  --protocol-artifact configs/hu-tuning-evaluation.json

.venv/bin/python -m poker.tuning_cli materialize \
  --config configs/ppo-sweep.json \
  --output-dir runs/tuning
```

Разрешён только небольшой allowlist PPO-параметров. Stage, model, league,
число рук на iteration, evaluation protocol и budget остаются общими. Для
каждой комбинации и seed создаётся стабильный trial id и готовый
`experiment.json`; процессы запускаются оператором независимо.

Запустить preregistered fixed evaluation для final full checkpoint:

```bash
.venv/bin/python -m poker.tuning_cli evaluate \
  --config configs/ppo-sweep.json \
  --trial-id trial-... \
  --checkpoint runs/.../checkpoints/periodic_00000010.pt \
  --output-dir runs/.../promotion
```

После неё связать результат с завершённым ledger:

```bash
.venv/bin/python -m poker.tuning_cli seal \
  --config configs/ppo-sweep.json \
  --trial-id trial-... \
  --checkpoint runs/.../checkpoints/periodic_00000010.pt \
  --ledger-manifest runs/.../experiment-ledger/experiment-manifest.json \
  --promotion-report runs/.../promotion/evaluations/evaluation_candidate_....json \
  --promotion-archive-manifest runs/.../promotion/archive/manifest.json \
  --report runs/.../sealed-evaluation.json
```

`seal` не выполняет evaluation и не принимает ручные score/ECE/pass flags. Он
проверяет terminal event завершённого experiment ledger, preregistered
protocol, source checkpoint и полный `PromotionEvaluator` report/archive,
после чего сам извлекает CI, ECE, illegal-action count и принятое решение.

Сравнение требует по одному завершённому report на каждый trial:

```bash
.venv/bin/python -m poker.tuning_cli compare \
  --config configs/ppo-sweep.json \
  --report runs/tuning/trial-a/fixed-evaluation.json \
  --report runs/tuning/trial-b/fixed-evaluation.json \
  --output runs/tuning/comparison.json
```

Перед ranking повторно проверяются full-checkpoint version/config/budget/SHA,
report SHA, protocol SHA и metric binding. Прошедшие trials сортируются по
нижней границе CI, затем по expected-showdown-share ECE и стабильному id.
Failed trial остаётся в отчёте без rank.

## Release registry

Только прошедший sealed report можно добавить в append-only registry. Все SHA
передаются явно, чтобы оператор видел точные release inputs:

```bash
.venv/bin/python -m poker.release_cli register \
  --registry releases \
  --release-id hu-a-2026-08 \
  --code-revision <commit-or-build-id> \
  --checkpoint runs/.../checkpoints/periodic_00000010.pt \
  --checkpoint-sha256 <sha256> \
  --ledger-manifest runs/.../experiment-ledger/experiment-manifest.json \
  --ledger-manifest-sha256 <sha256> \
  --tuning-report runs/.../sealed-evaluation.json \
  --tuning-report-sha256 <sha256>

.venv/bin/python -m poker.release_cli list --registry releases
.venv/bin/python -m poker.release_cli show --registry releases --release-id hu-a-2026-08
.venv/bin/python -m poker.release_cli verify --registry releases --release-id hu-a-2026-08
```

Registry повторно проверяет native full checkpoint, terminal ledger, protocol
artifact, promotion report/archive, derived CI/ECE и отсутствие illegal
actions. Model-only checkpoint, незавершённый trial, rejected report либо
изменённый lineage artifact отклоняются. Дополнительные pretraining/curriculum
артефакты можно закрепить повторяемым `--extra KIND=PATH=SHA256`.

Модель доверия здесь — operator-controlled local workspace. `code_revision`
является явно предоставленной внешней аттестацией commit/build, а не цифровой
подписью Git checkout. Hash-chain обнаруживает изменение уже опубликованных
артефактов, но не защищает от оператора, который до публикации намеренно
создал взаимно согласованный фиктивный набор файлов. Для adversarial registry
потребуется внешний CAS/signing service; он не входит в этот этап.

## Что этот этап не доказывает

В репозитории по-прежнему нет результатов длительного обучения. Наличие
ledger, sweep, comparison и release registry доказывает целостность
эксперимента, но не силу стратегии. Hyperparameters не меняются автоматически
во время run; новая комбинация всегда означает новый trial и новый seed matrix.
