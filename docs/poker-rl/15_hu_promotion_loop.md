# 15. Heads-up evaluation и promotion loop

Stage 2 запускает PPO от pretraining checkpoint и автоматически решает,
следует ли добавить новую frozen-политику в историческую лигу. Наличие
инфраструктуры не означает, что конкретная модель уже прошла gate: для этого
рядом с checkpoint должен существовать реальный evaluation report.

## Конфигурация

В обычный training config добавляются начальные веса и блок `promotion`:

```json
{
  "init_checkpoint": "runs/pretrain-stage-a/checkpoints/latest.pt",
  "promotion": {
    "enabled": true,
    "every_iterations": 5,
    "evaluate_on_complete": true,
    "hands_per_opponent": 200,
    "seed_start": 2000000,
    "equity_samples": 32,
    "calibration_bins": 10,
    "baseline_bots": ["rule", "tight", "aggro", "calling_station", "random"],
    "historical_limit": 4,
    "league_historical_limit": 8,
    "historical_weight": 1.0,
    "minimum_baseline_bb_per_100": 0.0,
    "minimum_baseline_ci95_low": -10000.0,
    "maximum_baseline_ci95_half_width": 10000.0,
    "minimum_historical_league_score": 0.45,
    "minimum_historical_ci95_low": -10000.0,
    "maximum_equity_ece": 0.08,
    "minimum_champion_improvement": 0.0
  }
}
```

Технические значения CI по умолчанию намеренно мягкие. Перед длительным run
их нужно зафиксировать по pilot-оценке и не менять между кандидатами.
Promotion сейчас разрешён только для heads-up stages A/B. Корректная
multiway expected-showdown-share метка уже реализована, но отдельный
multiway promotion protocol с heterogeneous seats ещё не введён.

## Протокол

1. После целостного PPO update runner атомарно сохраняет full-run checkpoint
   `checkpoints/candidate_<iteration>.pt`.
2. Из него создаётся immutable model-only candidate с SHA-256 в имени.
3. Каждый deal seed играется дважды: candidate занимает BTN и BB. CI считается
   по парным seed-блокам, а не по отдельным раздачам.
4. Candidate играет против фиксированного набора baseline и ограниченного
   списка promoted historical checkpoint'ов.
5. Gate проверяет BB/100 и границы CI, ширину CI, HU ECE, illegal actions,
   sanity scenarios, historical regression и строгое улучшение champion score.
6. Принятая модель копируется в immutable archive и только затем добавляется
   в opponent league. Отклонённый candidate и его report тоже сохраняются.

Champion score считается только на неизменном baseline-наборе и фиксированных
seed’ах. Historical opponents проверяются отдельно, поэтому рост лиги не
меняет шкалу сравнения новых champion’ов.

## Артефакты и восстановление

- `candidates/` — frozen snapshots всех проверенных кандидатов;
- `evaluations/` — machine-readable suite, candidate/source hashes, protocol,
  opponent registry и причины решения;
- `archive/` — принятые модели;
- `archive/manifest.json` — hashes reports/checkpoints, champion и вся история
  принятых и отклонённых решений;
- `checkpoints/promotion_<iteration>.pt` — full state после применения решения.

Файлы публикуются через fsync + atomic rename. Resume сверяет manifest с
report/checkpoint hashes. Если процесс завершился после записи archive manifest,
но до следующего full checkpoint, runner восстанавливает champion и promoted
league member из manifest. Evaluation обёрнут сохранением/restoration training
RNG, поэтому отклонённая проверка не меняет следующий rollout.

## Запуск

```bash
.venv/bin/python -m poker.train_cli \
  --config configs/hu-stage-a.json \
  --run-dir runs/hu-stage-a
```

После остановки продолжение остаётся обычным:

```bash
.venv/bin/python -m poker.train_cli \
  --run-dir runs/hu-stage-a \
  --resume latest
```

Любой promoted checkpoint можно открыть в mixed-policy UI через
`poker.watch` и сыграть против ботов или других checkpoint’ов.
