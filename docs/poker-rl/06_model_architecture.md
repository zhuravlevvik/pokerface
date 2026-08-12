# 06. Архитектура сети агента

## Результат

Одна PyTorch-модель, умеющая обрабатывать переменное число игроков и выдающая политику, размер ставки, value и equity.

## Backbone

```text
Own cards + board ──────> Card Encoder ───────┐
Per-player features ────> Set/Attention Encoder├─> Fusion MLP
Action-history tokens ──> Small Transformer ───┘
                                                   ├─ action head
                                                   ├─ bet-size head
                                                   ├─ value head
                                                   └─ equity head
```

- Карты — эмбеддинги ранга, масти и роли (private/public).
- Игроки — attention или permutation-invariant pooling с mask для отсутствующих и выбывших мест.
- История — токены `[street, actor position, action type, amount bucket]`; 3–6 Transformer-слоёв достаточно для MVP.
- Общая модель имеет общие веса на всех позициях за столом.

## Головы

- `action_head`: logits для `fold/check/call/raise`; нелегальные logits маскируются до softmax.
- `bet_size_head`: logits для дискретных рейз-сайзингов; используется лишь при `raise`.
- `value_head`: ожидаемый будущий PnL в BB.
- `equity_head`: softmax по `[win, tie, loss]`.

## Действия

1. Реализовать минимальную сеть и batched forward.
2. Добавить проверку форм, масок и конечности logits.
3. Написать детерминированный inference на фиксированном observation.
4. Сохранить checkpoint с версией observation и action-space.

## Критерий приёмки

Модель обрабатывает 2, 3 и 5 игроков без смены весов, возвращает нормированные распределения и никогда не выбирает замаскированное действие.
