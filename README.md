# Dialysis Simulation (Discrete Event Scheduling Research)

Research codebase for a discrete event simulation (DES) that evaluates patient scheduling heuristics in a resource-constrained dialysis unit.

## Project goals

- Compare scheduling strategies (e.g., FIFO vs fixed/priority policies) under identical stochastic scenarios.
- Run paired Monte Carlo experiments for statistically robust comparisons.
- Produce reproducible datasets and plots suitable for inclusion in research papers.

## Architecture (high level)

The code is organized as a modular pipeline:

- **Scenario generation**: stochastic shift scenarios (patients, resources, defects)
- **Strategy processing**: deterministic DES engine + pluggable strategies
- **Monte Carlo batching**: paired execution across strategies
- **Visualization**: statistical summaries + figures

See the spec at `../docs/superpowers/specs/2026-05-08-modular-des-pipeline-design.md`.

## Repository layout

- `main.py`: entrypoint (placeholder initially)
- `src/`: simulation modules
  - `scenario_generator.py`
  - `models.py`
  - `monte_carlo_batcher.py`
  - `visualizer.py`
  - `strategies/`: strategy implementations
- `tests/`: unit tests

## Getting started

Create a venv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Smoke check:

```bash
python -c "import dialysis_simulation.src"
```

## Papers / Results

This repository will accumulate research artifacts over time:

- `papers/` (planned): manuscripts, drafts, and supplementary material
- `results/` (planned): experiment outputs (not necessarily versioned)

