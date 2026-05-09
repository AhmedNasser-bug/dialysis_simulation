"""
Strategy Utilities Module

Helper classes, functions, and constants for procedural strategy execution.
Now incorporates "chair blocking" (bipartite separation) and nurse quotas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models import ShiftScenario, ShiftStatistics


MAX_TIME_SAFETY_LIMIT: int = 1_000_000
EXTREME_WAIT_PENALTY: float = 999999.0


@dataclass
class MachineState:
    id: int
    ready_time: int
    busy_until: int = 0
    assigned_patient_id: Optional[int] = None
    total_occupied_minutes: int = 0  # To track exact utilization


@dataclass
class NurseState:
    id: int
    busy_until: int = 0
    setups_completed: int = 0
    total_occupied_minutes: int = 0


@dataclass
class PatientSession:
    patient_id: int
    arrival_time: int
    setup_duration: int
    machine_reserved_minute: int = 0
    setup_start_time: int = 0
    session_start: int = 0
    session_end: int = 0
    wait_time: float = 0.0
    failed_to_serve: bool = False


def initialize_machine_states(scenario: ShiftScenario) -> Dict[int, MachineState]:
    machines: Dict[int, MachineState] = {}
    for mid, ready_time in scenario.machine_ready_times.items():
        if mid not in scenario.defective_machine_ids:
            # The machine is effectively "busy" (unavailable) until its ready_time
            machines[mid] = MachineState(id=mid, ready_time=ready_time, busy_until=ready_time)
    return machines


def initialize_nurse_states(nurse_count: int) -> List[NurseState]:
    return [NurseState(id=i) for i in range(1, nurse_count + 1)]


def find_earliest_available_nurse(nurses: List[NurseState], from_time: int) -> NurseState:
    """Finds the earliest available nurse at or after from_time, balancing setups equally."""
    # Sort by busy_until to find the one who is free earliest. 
    # If tied, prefer the nurse with fewer setups_completed to divide capacity equally.
    nurses.sort(key=lambda n: (max(from_time, n.busy_until), n.setups_completed))
    return nurses[0]


def calculate_wait_time(arrival_time: int, setup_start: int) -> float:
    return max(0.0, float(setup_start - arrival_time))


def aggregate_shift_statistics(
    strategy_name: str,
    scenario: ShiftScenario,
    patients: List[PatientSession],
    machines: Dict[int, MachineState],
    nurses: List[NurseState]
) -> ShiftStatistics:
    valid_waits = [p.wait_time for p in patients if not p.failed_to_serve]
    failed_count = sum(1 for p in patients if p.failed_to_serve)
    
    if valid_waits:
        mean_wait = sum(valid_waits) / len(valid_waits)
        max_wait = max(valid_waits)
    else:
        mean_wait = 0.0
        max_wait = 0.0
        
    shift_end = scenario.shift_end_minutes
    max_session_end = 0
    for p in patients:
        if not p.failed_to_serve:
            max_session_end = max(max_session_end, p.session_end)
            
    shift_overrun = max(0, max_session_end - shift_end)
    
    machine_utilization = 0.0
    if len(machines) > 0:
        total_machine_busy = sum(m.total_occupied_minutes for m in machines.values())
        machine_utilization = total_machine_busy / (len(machines) * shift_end)
        machine_utilization = min(1.0, machine_utilization)
        
    nurse_utilization = 0.0
    if len(nurses) > 0:
        total_nurse_busy = sum(n.total_occupied_minutes for n in nurses)
        nurse_utilization = total_nurse_busy / (len(nurses) * shift_end)
        nurse_utilization = min(1.0, nurse_utilization)
        
    return ShiftStatistics(
        strategy_name=strategy_name,
        total_patients_processed=len(patients),
        mean_wait_time_minutes=mean_wait,
        max_wait_time_minutes=max_wait,
        nurse_utilization_percent=nurse_utilization,
        machine_utilization_percent=machine_utilization,
        shift_overrun_minutes=shift_overrun,
        failed_patients_count=failed_count
    )
