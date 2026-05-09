
import pytest

from src.config import IntRange, SimulationConfig, UniformIntSampler
from src.models import ShiftScenario
from src.scenario.generator import generate_shift_scenario


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 123])
def test_patient_and_nurse_counts(seed: int) -> None:
    cfg = SimulationConfig(
        patient_volume=IntRange(3, 5),
        nurse_count=IntRange(2, 4),
    )

    scenario = generate_shift_scenario(cfg, seed=seed)

    assert 3 <= len(scenario.patient_arrivals) <= 5
    assert 2 <= scenario.nurse_count <= 4


@pytest.mark.parametrize("arrival_window_max", [0, 10, 60])
@pytest.mark.parametrize("seed", [0, 7, 99])
def test_arrival_times(arrival_window_max: int, seed: int) -> None:
    cfg = SimulationConfig(
        arrival_minute_sampler=UniformIntSampler(IntRange(0, arrival_window_max)),
    )

    scenario = generate_shift_scenario(cfg, seed=seed)

    for row in scenario.patient_arrivals:
        assert row["arrival_min"] >= 0
        assert row["arrival_min"] <= arrival_window_max


@pytest.mark.parametrize("max_machine_ready_delay", [0, 15, 90])
@pytest.mark.parametrize("seed", [0, 3, 55])
def test_machine_readiness(max_machine_ready_delay: int, seed: int) -> None:
    cfg = SimulationConfig(
        machine_defect_probability=0.0,
        total_machines=IntRange(5, 5),
        machine_ready_delay_minutes=IntRange(0, max_machine_ready_delay),
    )
    scenario = generate_shift_scenario(cfg, seed=seed)

    # With defect prob forced to 0.0, all machines should be present.
    assert len(scenario.machine_ready_times) == 5

    for ready_minute in scenario.machine_ready_times.values():
        assert 0 <= ready_minute <= max_machine_ready_delay


def test_defect_logic() -> None:
    total_machines = 6
    seed = 123

    cfg_all_defect = SimulationConfig(
        machine_defect_probability=1.0,
        total_machines=IntRange(total_machines, total_machines),
    )
    scenario_all_defect = generate_shift_scenario(cfg_all_defect, seed=seed)

    assert scenario_all_defect.defective_machine_ids == list(range(1, total_machines + 1))
    assert scenario_all_defect.machine_ready_times == {}

    cfg_no_defect = SimulationConfig(
        machine_defect_probability=0.0,
        total_machines=IntRange(total_machines, total_machines),
    )
    scenario_no_defect = generate_shift_scenario(cfg_no_defect, seed=seed)

    assert scenario_no_defect.defective_machine_ids == []
    assert sorted(scenario_no_defect.machine_ready_times.keys()) == list(range(1, total_machines + 1))


def test_schema_integrity() -> None:
    cfg = SimulationConfig()
    scenario = generate_shift_scenario(cfg, seed=999)

    assert isinstance(scenario, ShiftScenario)

    # Required fields populated with correct types/shapes
    assert isinstance(scenario.patient_arrivals, list)
    assert len(scenario.patient_arrivals) > 0
    assert isinstance(scenario.nurse_count, int)
    assert isinstance(scenario.machine_ready_times, dict)
    assert isinstance(scenario.defective_machine_ids, list)
    assert scenario.scenario_seed == 999

    # Patient arrival rows contain expected keys
    for row in scenario.patient_arrivals:
        assert set(row.keys()) == {"id", "arrival_min", "setup_min", "session_min"}
        assert isinstance(row["id"], int)
        assert isinstance(row["arrival_min"], int)
        assert isinstance(row["setup_min"], int)
