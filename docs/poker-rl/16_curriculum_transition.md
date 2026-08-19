# 16. Автоматический переход curriculum A → B

## Назначение и границы

Этот механизм — **opt-in** автоматизация только для heads-up перехода от
Stage A (сокращённые сайзинги) к Stage B (полный набор сайзингов). Он не
утверждает, что модель сильная сама по себе: переход разрешается лишь по
зафиксированному evaluation-протоколу и оставляет проверяемые артефакты.

Этот встроенный механизм остаётся только A→B и сам не содержит scratch-arm.
Новый отдельный coordinator с настоящим paired rung и multiway gate описан в
[17_paired_multiway_curriculum.md](17_paired_multiway_curriculum.md). Он не
расширяет mutable stage этого legacy run, а создаёт отдельные target-stage
segments с full checkpoints.

## Конфигурация и запуск

Переход выключен по умолчанию. Его можно включать только в новом Stage A run;
в том же run нельзя включать `promotion`.

```json
{
  "run": {"stage": "A"},
  "curriculum": {
    "base_learning_rate": 0.0003,
    "require_transfer_beats_scratch": false
  },
  "promotion": {"enabled": false},
  "transition": {
    "enabled": true,
    "source_stage": "A",
    "target_stage": "B",
    "every_iterations": 5,
    "evaluate_on_complete": true,
    "hands_per_opponent": 200,
    "seed_start": 3000000,
    "equity_samples": 32,
    "baseline_bots": ["rule", "tight", "aggro", "calling_station", "random"],
    "minimum_baseline_ci95_low": 0.0,
    "maximum_baseline_ci95_half_width": 20.0,
    "minimum_prior_ci95_low": 0.0,
    "reference_checkpoint": "runs/hu-a-reference/checkpoints/latest.pt",
    "reference_checkpoint_sha256": "<64 lowercase hex characters>",
    "reset_optimizer": true
  }
}
```

`reference_checkpoint` обязателен и задаётся оператором явно вместе с
`reference_checkpoint_sha256`. Это immutable **full-run checkpoint Stage A**
(например source checkpoint, связанный с принятым promotion report), с которым
проверяется отсутствие регрессии. Runner проверяет stage, формат, SHA-256,
совместимость модели и отличие весов reference от кандидата. Нельзя заменять
его на «текущую модель по умолчанию» или менять между retry одной и той же
итерации.

`transition.curriculum` (если указан) должен в точности совпадать с верхним
`curriculum`. Автоматический A→B требует
`require_transfer_beats_scratch=false`: это техническое признание отсутствия
paired scratch rung, а не ослабление научного требования. Следовательно,
принятый автоматический переход **не доказывает** преимущество transfer над
scratch; такое сравнение проводится отдельным экспериментом до утверждений о
качестве transfer.

Обычный запуск остаётся прежним:

```bash
.venv/bin/python -m poker.train_cli \
  --config configs/hu-stage-a-to-b.json \
  --run-dir runs/hu-a-to-b
```

## Что происходит на границе

После целостного PPO update, когда наступила настроенная граница (или конец
run при `evaluate_on_complete=true`), runner:

1. сохраняет full-run source checkpoint;
2. замораживает immutable A-candidate и запускает парную HU evaluation против
   baseline и обязательного reference checkpoint;
3. сохраняет report, hash'и, конфигурацию и решение;
4. при принятии публикует model-only transfer checkpoint Stage B, переводит
   scheduler на B и создаёт новый optimizer;
5. применяет learning rate Stage B: `base_learning_rate × 0.7`.

При `reset_optimizer=true` optimizer state намеренно не переносится из A.
Сохраняются веса модели и полный execution progress run, но moments optimizer
начинаются заново уже с LR Stage B. При `false` moments сохраняются, однако
stage-aware LR всё равно становится `×0.7`.

Если gate отклонён, stage, optimizer и поток следующего rollout не меняются.
Evaluation восстанавливает RNG обучающего процесса после своей работы.

## Неизменяемые доказательства и resume

В `runs/hu-a-to-b/curriculum-transitions/` публикуются через `fsync` и atomic
rename:

- `candidates/` — frozen source snapshot с SHA-256;
- `reports/` — machine-readable evidence: config/run hashes, seed protocol,
  матчапы, CI, calibration, sanity/illegal-action проверки, причины решения;
- `transfers/` — только принятый model-only Stage B artifact;
- `manifest.json` — связанная hash-проверяемая история принятых и отклонённых
  решений.

Full-run checkpoint хранит transition state и SHA manifest. При `--resume`
runner сверяет эти артефакты до продолжения. Если `Ctrl+C` произошёл после
границы PPO, но до transition evaluation, или между записью evidence и
следующим full checkpoint, resume завершает тот же pending transition до
нового rollout. Повторная оценка одной итерации идемпотентна: она должна
сослаться на уже опубликованное решение, а не создать другое.

Первый `Ctrl+C` по-прежнему является graceful shutdown, второй — немедленной
остановкой. Не удаляйте candidate/report/manifest вручную: при нарушении hash
или provenance resume завершится fail-closed.

## Приёмка A → B

Принятый переход должен иметь один проверяемый manifest decision, report и
transfer artifact с согласованными hash'ами. Evidence включает:

- положительный/заданный baseline результат с требуемой нижней границей CI;
- prior/reference matchup и его CI;
- expected-showdown-share calibration, sanity scenarios и ноль illegal actions;
- `StageEvaluation`, совместимую с `CurriculumConfig` при явно отключённом
  scratch-сравнении.

После перехода необходимо отдельно посмотреть transfer checkpoint в
`poker.watch`, а заявление «transfer лучше scratch» допускается только после
реального paired scratch эксперимента.
