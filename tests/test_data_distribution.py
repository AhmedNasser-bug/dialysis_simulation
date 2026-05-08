from __future__ import annotations

from collections import Counter

import pytest

from dialysis_simulation.src.config import IntRange, SimulationConfig, UniformIntSampler
from dialysis_simulation.src.scenario_generator import generate_shift_scenario


NUM_SCENARIOS = 10_000


def test_machine_defect_rate_converges_over_10000_scenarios() -> None:
    """
    Generate 10,000 scenarios and verify empirical defect rate converges to
    configured probability within a ±2% absolute margin.
    """

    cfg = SimulationConfig(
        machine_defect_probability=0.15,
        total_machines=8,
    )

    defective_count = 0
    total_machine_observations = NUM_SCENARIOS * cfg.total_machines

    for seed in range(NUM_SCENARIOS):
        scenario = generate_shift_scenario(cfg, seed=seed)
        defective_count += len(scenario.defective_machine_ids)

    observed_defect_rate = defective_count / total_machine_observations
    assert observed_defect_rate == pytest.approx(cfg.machine_defect_probability, abs=0.02)


def test_fixed_patient_volume_is_deterministic_over_10000_scenarios() -> None:
    """
    Edge case: min_patients == max_patients must yield strict deterministic
    patient counts over many runs.
    """

    cfg = SimulationConfig(patient_volume=IntRange(4, 4))

    for seed in range(NUM_SCENARIOS):
        scenario = generate_shift_scenario(cfg, seed=seed)
        assert len(scenario.patient_arrivals) == 4


@pytest.mark.parametrize("defect_probability", [0.0, 1.0])
def test_defect_probability_extremes(defect_probability: float) -> None:
    """
    Edge cases:
    - defect_probability=0.0 => no defective machines
    - defect_probability=1.0 => all machines defective
    """

    cfg = SimulationConfig(
        machine_defect_probability=defect_probability,
        total_machines=6,
    )
    expected_defective = 0 if defect_probability == 0.0 else cfg.total_machines

    for seed in range(1_000):
        scenario = generate_shift_scenario(cfg, seed=seed)
        assert len(scenario.defective_machine_ids) == expected_defective


def test_setup_times_fit_uniform_distribution_over_10000_samples() -> None:
    """
    Validate setup durations follow a discrete uniform draw over [10, 20].

    We collect exactly 10,000 setup samples (one patient per scenario) and
    apply a chi-square goodness-of-fit check against equal-frequency bins.
    """

    setup_range = IntRange(10, 20)
    cfg = SimulationConfig(
        patient_volume=IntRange(1, 1),
        setup_duration_minutes_sampler=UniformIntSampler(setup_range),
    )

    setup_counts: Counter[int] = Counter()
    total_setup_samples = 0

    for seed in range(NUM_SCENARIOS):
        scenario = generate_shift_scenario(cfg, seed=seed)
        setup_min = scenario.patient_arrivals[0]["setup_min"]
        setup_counts[setup_min] += 1
        total_setup_samples += 1

    expected_values = list(range(setup_range.low, setup_range.high + 1))
    assert sorted(setup_counts.keys()) == expected_values

    expected_count_per_bin = total_setup_samples / len(expected_values)
    chi_square_stat = sum(
        ((setup_counts[value] - expected_count_per_bin) ** 2) / expected_count_per_bin
        for value in expected_values
    )

    # 11 bins => df=10, with 99.9th percentile chi-square ≈ 29.588
    assert chi_square_stat < 29.588
