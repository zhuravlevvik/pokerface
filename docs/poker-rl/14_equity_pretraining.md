# 14. Предобучение equity/backbone

Stage 1 создаёт воспроизводимый restricted corpus с player-safe model inputs, обучает backbone и
equity-head до PPO и сохраняет отчёт, по которому можно решить, готова ли
инициализация к heads-up self-play.

## Первый запуск

Создать стартовый конфиг:

```bash
.venv/bin/python -m poker.pretrain_cli \
  --write-default-config configs/pretrain-stage-a.json
```

Запустить corpus generation и обучение:

```bash
.venv/bin/python -m poker.pretrain_cli \
  --config configs/pretrain-stage-a.json \
  --run-dir runs/pretrain-stage-a
```

Конфиг фиксирует непересекающиеся диапазоны seed для train/holdout, смесь
baseline-политик, число Monte-Carlo досдач, архитектуру, optimizer, число
epoch и acceptance gates. По умолчанию поведенческое клонирование и value
warm-up выключены: основной loss — soft cross-entropy equity-head.

Исполняемый pretraining runner допускает только stages A/B. Категории
`win/tie/loss` осмысленны и в multiway, но текущий scalar calibration
`win + 0.5 × tie` является ожидаемой долей банка только в heads-up.

## Артефакты

В run directory создаются:

- `corpus.jsonl` — безопасные observation/labels и audit metadata без карт
  оппонентов, undealt deck или private snapshots;
- `checkpoints/*.pt` и `latest.pt` — model, optimizer, epoch/step, RNG,
  config/corpus hashes и label protocol;
- `report.json` — logloss, multiclass Brier, ECE, heads-up MAE/RMSE и
  сравнение с train-only empirical prior в разрезах
  street × players × active opponents;
- `manifest.json` — список опубликованных checkpoint'ов.

Метка имеет явную семантику `fixed_deal_virtual_showdown_v1`: при её
построении известны фактически сданные карты оппонентов. Эти карты никогда
не входят в observation или JSONL, но target не следует называть range equity.
Audit `hand_seed` хранится вне observation и позволяет оператору полностью
воспроизвести сдачу, поэтому corpus — внутренний training artifact: его нельзя
публиковать клиенту и нельзя передавать seed в модель.

## Остановка и продолжение

Первый `Ctrl+C` просит завершить текущую epoch, после чего атомарно пишется
`interrupt_<epoch>.pt`. Второй `Ctrl+C` останавливает процесс немедленно без
публикации частично обновлённого состояния.

Продолжить последний checkpoint:

```bash
.venv/bin/python -m poker.pretrain_cli \
  --run-dir runs/pretrain-stage-a \
  --resume latest
```

Или задать конкретный checkpoint и абсолютную целевую epoch:

```bash
.venv/bin/python -m poker.pretrain_cli \
  --resume runs/pretrain-stage-a/checkpoints/interrupt_00000004.pt \
  --epochs 10
```

Resume проверяет SHA-256 corpus/config до загрузки optimizer. Изменённый или
подменённый corpus отвергается, а следующий порядок minibatch и RNG
продолжают исходный детерминированный поток на протестированном CPU/runtime.
Для bit-for-bit обещания между разными CUDA-версиями нужен отдельный
deterministic-algorithms режим и собственная проверка.

## Переход к PPO

`report.json.acceptance.passed` — предварительный gate, а не доказательство
силы стратегии. После его прохождения `latest.pt` можно передать обычному
model loader и использовать как старт Stage A PPO. Затем отдельно нужны
позиционно ротированные матчи против baseline, BB/100 с доверительными
интервалами и regression control set.
