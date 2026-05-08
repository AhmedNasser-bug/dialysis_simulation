# Contributing

This is a research-focused codebase. Contributions should prioritize **reproducibility**, **clarity**, and **traceability**.

## Principles

- **Determinism**: given the same seed + scenario, the strategy processor must produce identical results.
- **Separation of concerns**: keep scenario generation stochastic; keep strategy processing deterministic.
- **Explicit configuration**: avoid hard-coded constants in simulation logic.

## Workflow

- Create a focused change (one behavior or one refactor at a time).
- Add or update tests in `tests/` when behavior changes.
- Prefer small, reviewable commits with clear messages.

## Testing

Run tests:

```bash
pytest
```

## Research artifacts

When we add papers later, please avoid committing large generated datasets/binaries unless explicitly intended. If you introduce new output directories (e.g., `results/`), add appropriate ignore rules.

