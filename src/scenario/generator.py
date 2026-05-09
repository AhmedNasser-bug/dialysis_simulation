"""
Module 1: Stochastic Scenario Creator

This module is responsible for generating the initial conditions of a *single*
dialysis-unit shift under uncertainty (arrivals, staffing levels, readiness
delays, and pre-shift defects).

Important:
- This module does NOT simulate the progression of time.
- It does NOT apply scheduling strategies.
- It only creates a deterministic ShiftScenario given (config, seed).

Downstream modules (strategy processor + Monte Carlo batcher) can reuse the same
ShiftScenario across multiple strategies to support paired-difference testing.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List

from src.config import SimulationConfig
from src.models import ShiftScenario


@dataclass(frozen=True, slots=True)
class MachineModel:
    """
    Local scenario-generation helper (not part of the public scenario schema).

    Keeping this small object makes the generation logic easier to test and
    reason about without leaking implementation details into ShiftScenario.
    """

    machine_id: int
    ready_minute: int
    is_defective: bool


def generate_shift_scenario(
    config: SimulationConfig,
    *,
    seed: int = 0,
) -> ShiftScenario:
    """
    Generate a single ShiftScenario from a SimulationConfig.

    Parameters
    ----------
    config:
        Centralized parameter set. Stochastic values are sampled from its ranges
        and sampler callables.
    seed:
        Seed for deterministic scenario creation.
    Notes:
        Machine session parameters (total machines and machine readiness delay
        range) are sourced from SimulationConfig so scenario setup remains fully
        centralized in one config object.

    Returns
    -------
    ShiftScenario
        Immutable snapshot of the shift's initial constraints.
    """

    config.validate()
    rng = random.Random(seed)

    patient_count = config.sample_patient_count(rng)
    nurse_count = config.sample_nurse_count(rng)

    # Generate patient arrival rows (stochastic, but deterministic under seed).
    patient_arrivals: List[Dict[str, int]] = []
    for patient_id in range(1, patient_count + 1):
        arrival_min = config.sample_arrival_minute(rng)
        setup_min = config.sample_setup_duration_minutes(rng)
        session_min = config.sample_session_duration_minutes(rng)
        patient_arrivals.append({
            "id": patient_id, 
            "arrival_min": arrival_min, 
            "setup_min": setup_min,
            "session_min": session_min
        })

    # Machines: readiness delays + pre-shift binary defect test at T=0.
    total_machines = config.total_machines.sample(rng)

    machines: List[MachineModel] = []
    for machine_id in range(1, total_machines + 1):
        ready_minute = config.sample_machine_ready_delay_minutes(rng)
        is_defective = config.sample_machine_is_defective(rng)
        machines.append(MachineModel(machine_id=machine_id, ready_minute=ready_minute, is_defective=is_defective))

    defective_machine_ids = [m.machine_id for m in machines if m.is_defective]
    machine_ready_times: Dict[int, int] = {
        m.machine_id: m.ready_minute for m in machines if not m.is_defective
    }

    # Note: defective machines are removed from the active pool by omission from
    # machine_ready_times and explicit listing in defective_machine_ids.
    return ShiftScenario(
        patient_arrivals=patient_arrivals,
        nurse_count=nurse_count,
        machine_ready_times=machine_ready_times,
        defective_machine_ids=defective_machine_ids,
        scenario_seed=seed,
        machine_cooldown_minutes=config.machine_cooldown_minutes,
        shift_end_minutes=config.shift_duration_minutes,
    )
