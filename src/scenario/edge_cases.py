from typing import Dict
import random

from src.config import SimulationConfig, UniformIntSampler
from src.models import ShiftScenario

def generate_burst_arrival(config: SimulationConfig, seed: int = 42) -> ShiftScenario:
    rng = random.Random(seed)
    patient_count = config.patient_volume.high
    nurse_count = config.nurse_count.low

    patient_arrivals = []
    for pid in range(1, patient_count + 1):
        patient_arrivals.append({
            "id": pid,
            "arrival_min": 0,  # Everyone arrives at T=0
            "setup_min": config.sample_setup_duration_minutes(rng),
            "session_min": config.sample_session_duration_minutes(rng)
        })

    total_machines = config.total_machines.high
    machine_ready_times = {mid: 0 for mid in range(1, total_machines + 1)}

    return ShiftScenario(
        patient_arrivals=patient_arrivals,
        nurse_count=nurse_count,
        machine_ready_times=machine_ready_times,
        defective_machine_ids=[],
        scenario_seed=seed,
        machine_cooldown_minutes=config.machine_cooldown_minutes,
        shift_end_minutes=config.shift_duration_minutes,
        min_session_duration_minutes=config.min_session_duration_minutes,
    )

def generate_maintenance_disaster(config: SimulationConfig, seed: int = 42) -> ShiftScenario:
    rng = random.Random(seed)
    patient_count = config.patient_volume.high
    nurse_count = config.nurse_count.high

    patient_arrivals = []
    for pid in range(1, patient_count + 1):
        patient_arrivals.append({
            "id": pid,
            "arrival_min": config.sample_arrival_minute(rng),
            "setup_min": config.sample_setup_duration_minutes(rng),
            "session_min": config.sample_session_duration_minutes(rng)
        })

    total_machines = config.total_machines.low
    machine_ready_times = {mid: 0 for mid in range(1, total_machines + 1)}

    # 80% of machines are defective
    defective_count = int(total_machines * 0.8)
    defective_machine_ids = list(range(1, defective_count + 1))

    return ShiftScenario(
        patient_arrivals=patient_arrivals,
        nurse_count=nurse_count,
        machine_ready_times=machine_ready_times,
        defective_machine_ids=defective_machine_ids,
        scenario_seed=seed,
        machine_cooldown_minutes=config.machine_cooldown_minutes,
        shift_end_minutes=config.shift_duration_minutes,
        min_session_duration_minutes=config.min_session_duration_minutes,
    )

def generate_marathon_shift(config: SimulationConfig, seed: int = 42) -> ShiftScenario:
    patient_count = config.patient_volume.high
    nurse_count = config.nurse_count.low

    # Try to extract the max bounds, fallback to reasonable large numbers
    max_setup = 30
    if isinstance(config.setup_duration_minutes_sampler, UniformIntSampler):
        max_setup = config.setup_duration_minutes_sampler.minute_range.high

    patient_arrivals = []
    for pid in range(1, patient_count + 1):
        patient_arrivals.append({
            "id": pid,
            "arrival_min": pid * 10,
            "setup_min": max_setup,
            "session_min": config.session_duration_minutes_range.high
        })

    total_machines = config.total_machines.low
    machine_ready_times = {mid: 0 for mid in range(1, total_machines + 1)}

    return ShiftScenario(
        patient_arrivals=patient_arrivals,
        nurse_count=nurse_count,
        machine_ready_times=machine_ready_times,
        defective_machine_ids=[],
        scenario_seed=seed,
        machine_cooldown_minutes=config.machine_cooldown_minutes,
        shift_end_minutes=config.shift_duration_minutes,
        min_session_duration_minutes=config.min_session_duration_minutes,
    )

def get_all_edge_cases(config: SimulationConfig) -> Dict[str, ShiftScenario]:
    return {
        "Burst Arrival": generate_burst_arrival(config),
        "Maintenance Disaster": generate_maintenance_disaster(config),
        "Marathon Shift": generate_marathon_shift(config)
    }
