# 10. Оценка качества и диагностика

## Результат

Воспроизводимый evaluation-suite, который отделяет реальный прогресс от удачной серии раздач.

## Протокол

- Фиксированный набор seed’ов и число рук.
- Равная ротация позиций.
- Оценка против каждого baseline и ключевых historical checkpoint’ов.
- Checkpoint оценивается отдельно от данных, использованных для обучения.

## Метрики

```text
Стратегия: BB/100, win rate, Elo/league score.
Покерные статусы: VPIP, PFR, 3-bet, fold-to-3-bet, aggression factor.
Обучение: policy entropy, value error, доля masked действий.
Equity: cross-entropy, Brier score, ECE и reliability diagram.
```

## Sanity checks

Создать фиксированный набор позиций: натс на ривере, заведомо проигрышная рука, флеш-дро, выгодные и невыгодные pot odds, short-stack all-in. Проверять, что модель не принимает абсурдных решений и не выдаёт некалиброванную equity.

## Критерий приёмки

Каждый checkpoint имеет machine-readable отчёт; решения о promotion основаны на фиксированных метриках, а не отдельных красивых раздачах.
