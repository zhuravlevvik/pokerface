# Stage A: multi-seed smoke и tuning campaign

Одиночный успешно прошедший trial не является доказательством качества:
результат может быть удачным выбросом конкретного training seed. Campaign
объединяет полный sweep по одинаковому набору seed и ранжирует только варианты
гиперпараметров, для которых доступны все preregistered evidence artifacts.

## Семантика сравнения

- training seed меняет обучение, но не evaluation: `evaluation_run_seed=0`
  закреплён внутри HU protocol artifact;
- warm-start checkpoint входит в run/trial identity вместе с SHA-256 и kind;
  pretraining warm-start дополнительно требует hash-pinned evidence report;
- каждый trial сначала проходит собственные PromotionEvaluator gates;
- для каждого baseline отдельно считается средний `BB/100` по training seed и
  двухсторонний 95% Student-t CI;
- вариант проходит только если все seed-trials прошли, максимальный ECE не
  выше campaign ceiling и нижняя CI-граница каждого baseline выше floor;
- требуется минимум два seed. Один seed не получает rank и не может стать
  campaign winner;
- итоговый JSON повторно открывает terminal checkpoints, ledger hash-chain,
  sealed reports, promotion reports/archives и protocol hashes.

Это сравнение выбирает устойчивый вариант гиперпараметров. Оно не выбирает
произвольный «лучший seed checkpoint» для production release. После выбора
варианта нужен отдельный canonical training run с заранее заданным seed и
обычной release evaluation.

## Полная операторская цепочка

Сначала создайте обычный `SweepConfig` через `poker.tuning_cli`, затем один раз
зафиксируйте campaign gates:

```bash
.venv/bin/python -m poker.campaign_cli init \
  --sweep-config configs/ppo-sweep.json \
  --output configs/stage-a-campaign.json \
  --minimum-seeds 3 \
  --minimum-baseline-ci95-low 0 \
  --maximum-ece 0.08
```

В конфиге закрепляются полный sweep, `minimum_seeds_per_variant`, cross-seed CI
floor и ECE ceiling.

Дальше все пути выводятся из materialized trial id:

```bash
.venv/bin/python -m poker.campaign_cli run \
  --config configs/stage-a-campaign.json \
  --trials-dir runs/stage-a-campaign/trials \
  --runs-dir runs/stage-a-campaign/runs

.venv/bin/python -m poker.campaign_cli status \
  --config configs/stage-a-campaign.json \
  --runs-dir runs/stage-a-campaign/runs

.venv/bin/python -m poker.campaign_cli evaluate-seal \
  --config configs/stage-a-campaign.json \
  --trials-dir runs/stage-a-campaign/trials \
  --runs-dir runs/stage-a-campaign/runs \
  --evidence-dir runs/stage-a-campaign/evidence

.venv/bin/python -m poker.campaign_cli aggregate \
  --config configs/stage-a-campaign.json \
  --evidence runs/stage-a-campaign/evidence/trial-a/sealed.json \
  --evidence runs/stage-a-campaign/evidence/trial-b/sealed.json \
  --output runs/stage-a-campaign/campaign-report.json

.venv/bin/python -m poker.campaign_cli verify \
  --config configs/stage-a-campaign.json \
  --report runs/stage-a-campaign/campaign-report.json
```

Первый `Ctrl+C` внутри `run` останавливает текущий trial на целой PPO boundary.
Повтор той же команды продолжает его, затем идёт к следующему trial. Неполный,
missing или `failed_nonfinite` trial останавливает кампанию и не допускается к
aggregation.

## Health summary

Для одного trial можно получить read-only player-safe сводку, не читая tensor
checkpoint и не доверяя заменяемому `metrics.jsonl`:

```bash
.venv/bin/python -m poker.experiment_summary_cli \
  --run-dir runs/stage-a-campaign/runs/trial-... \
  --output runs/stage-a-campaign/runs/trial-.../health.json \
  --max-abs-kl 0.2 \
  --max-clip-fraction 0.5
```

Пороги health по умолчанию отсутствуют: summary показывает min/max/last и
явные alerts, но не подменяет promotion/campaign gate.

## Локальный smoke, выполненный при интеграции

Это не committed evidence и не критерий release. Локальный интеграционный
smoke прошёл полную одиночную цепочку на Stage A: 2 PPO
iterations, 4 hands, 12 decisions, затем fixed evaluation и sealing. Все
значения остались finite, illegal actions — 0. Полученная нижняя граница
`-4950 BB/100` не интерпретируется как качество: evaluation содержала всего две
раздачи и проверяла только работоспособность artifact pipeline.

Перед реальной кампанией увеличьте число training iterations/hands и число
paired evaluation blocks, зафиксируйте минимум 3–5 training seed и не меняйте
пороги после просмотра результатов.
