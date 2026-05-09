"""
Strategy Utilities Module

This module contains all helper classes, functions, and constants used by
scheduling strategies. It decouples implementation details from the strategy
logic, promoting code reuse and maintainability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models import ShiftScenario


# =============================================================================
# Constants (shared across all strategies)
# =============================================================================

MAX_TIME_SAFETY_LIMIT: int = 1_000_000
"""Safety limit for simulation time to prevent infinite loops."""

EXTREME_WAIT_PENALTY: float = 999999.0
"""Penalty value for patients who cannot be served (e.g., defective machine)."""


# =============================================================================
# Internal State Tracking Classes
# =============================================================================

@dataclass
class MachineState:
    """
    Internal state tracker for a machine during strategy execution.
    
    Attributes:
        id: Unique machine identifier.
        ready_time: Minute when the machine becomes ready for use.
        busy_until: Minute when the machine will be free (after setup + session + cooldown).
        assigned_patient_id: ID of patient currently assigned to this machine (if any).
    """
    id: int
    ready_time: int
    busy_until: int = 0
    assigned_patient_id: Optional[int] = None


@dataclass
class NurseState:
    """
    Internal state tracker for a nurse during strategy execution.
    
    Attributes:
        id: Unique nurse identifier.
        busy_until: Minute when the nurse will be free (after setup completion).
    """
    id: int
    busy_until: int = 0


@dataclass
class PatientSession:
    """
    Tracks a patient's session timing and outcomes.
    
    Attributes:
        patient_id: Unique patient identifier.
        arrival_time: Minute when the patient arrives at the facility.
        setup_duration: Minutes required for nurse setup.
        session_start: Minute when the dialysis session begins.
        session_end: Minute when the dialysis session ends.
        wait_time: Minutes the patient waited before session start.
    """
    patient_id: int
    arrival_time: int
    setup_duration: int
    session_start: int = 0
    session_end: int = 0
    wait_time: float = 0.0


# =============================================================================
# Resource Availability Functions
# =============================================================================

def get_available_nurse(nurses: List[NurseState], at_time: int) -> Optional[NurseState]:
    """
    Find a nurse that is available at the given time.
    
    Args:
        nurses: List of nurse state objects.
        at_time: The time at which to check availability.
        
    Returns:
        A NurseState object if one is available, otherwise None.
    """
    for nurse in nurses:
        if nurse.busy_until <= at_time:
            return nurse
    return None


def find_earliest_nurse_availability(nurses: List[NurseState], from_time: int) -> int:
    """
    Find the earliest time >= from_time when any nurse becomes available.
    
    Args:
        nurses: List of nurse state objects.
        from_time: The starting time to search from.
        
    Returns:
        The earliest time when a nurse is available.
    """
    # Check if any nurse is already available
    for nurse in nurses:
        if nurse.busy_until <= from_time:
            return from_time
    
    # Find the earliest future availability
    earliest = float('inf')
    for nurse in nurses:
        if nurse.busy_until < earliest:
            earliest = nurse.busy_until
    
    return earliest if earliest != float('inf') else from_time


def initialize_machine_states(scenario: ShiftScenario) -> Dict[int, MachineState]:
    """
    Initialize machine states from a scenario, excluding defective machines.
    
    Args:
        scenario: The shift scenario containing machine information.
        
    Returns:
        Dictionary mapping machine IDs to MachineState objects.
    """
    machines: Dict[int, MachineState] = {}
    for mid, ready_time in scenario.machine_ready_times.items():
        if mid not in scenario.defective_machine_ids:
            machines[mid] = MachineState(id=mid, ready_time=ready_time)
    return machines


def initialize_nurse_states(nurse_count: int) -> List[NurseState]:
    """
    Initialize nurse states for a given number of nurses.
    
    Args:
        nurse_count: Number of nurses in the shift.
        
    Returns:
        List of NurseState objects.
    """
    return [NurseState(id=i) for i in range(nurse_count)]


# =============================================================================
# Timing Calculation Functions
# =============================================================================

def calculate_session_end(session_start: int, setup_duration: int, 
                          session_duration: int) -> int:
    """
    Calculate when a patient's session ends.
    
    Args:
        session_start: Minute when the session starts.
        setup_duration: Minutes required for setup.
        session_duration: Duration of the actual dialysis session.
        
    Returns:
        Minute when the session ends (setup + session duration).
    """
    return session_start + setup_duration + session_duration


def calculate_machine_busy_until(session_end: int, cooldown: int) -> int:
    """
    Calculate when a machine becomes free after a session.
    
    Args:
        session_end: Minute when the session ends.
        cooldown: Cooldown period after session.
        
    Returns:
        Minute when the machine is free again.
    """
    return session_end + cooldown


def calculate_wait_time(arrival_time: int, session_start: int) -> float:
    """
    Calculate patient wait time.
    
    Args:
        arrival_time: Minute when patient arrived.
        session_start: Minute when session started.
        
    Returns:
        Wait time in minutes (non-negative).
    """
    return max(0.0, float(session_start - arrival_time))


def calculate_overrun(session_end: int, shift_end: int) -> int:
    """
    Calculate shift overrun for a single session.
    
    Args:
        session_end: Minute when the session ends.
        shift_end: Official shift end time.
        
    Returns:
        Overrun minutes (0 if no overrun).
    """
    overrun = session_end - shift_end
    return max(0, overrun)


# =============================================================================
# Utilization Calculation Functions
# =============================================================================

def calculate_nurse_utilization(patients: List[PatientSession], 
                                 nurse_count: int, 
                                 max_time: int) -> float:
    """
    Calculate average nurse utilization as a fraction [0, 1].
    
    Args:
        patients: List of patient sessions.
        nurse_count: Total number of nurses.
        max_time: Maximum time horizon for utilization calculation.
        
    Returns:
        Nurse utilization as a fraction between 0 and 1.
    """
    if max_time <= 0 or max_time == float('inf'):
        return 0.0
    
    total_setup_time = sum(
        p.setup_duration for p in patients 
        if p.session_start < float('inf')
    )
    utilization = total_setup_time / (nurse_count * max_time) if max_time > 0 else 0.0
    return min(1.0, utilization)


def calculate_machine_utilization(patients: List[PatientSession],
                                   num_machines: int,
                                   max_time: int,
                                   session_duration: int) -> float:
    """
    Calculate average machine utilization as a fraction [0, 1].
    
    Args:
        patients: List of patient sessions.
        num_machines: Total number of available machines.
        max_time: Maximum time horizon for utilization calculation.
        session_duration: Duration of each dialysis session.
        
    Returns:
        Machine utilization as a fraction between 0 and 1.
    """
    if max_time <= 0 or max_time == float('inf') or num_machines <= 0:
        return 0.0
    
    total_session_time = sum(
        session_duration for p in patients
        if p.session_start < float('inf')
    )
    utilization = total_session_time / (num_machines * max_time)
    return min(1.0, utilization)


def calculate_max_time(patients: List[PatientSession]) -> int:
    """
    Find the maximum session end time across all patients.
    
    Args:
        patients: List of patient sessions.
        
    Returns:
        Maximum session end time in minutes.
    """
    max_time = 0
    for patient in patients:
        if patient.session_end < float('inf'):
            max_time = max(max_time, patient.session_end)
    return max_time


# =============================================================================
# Statistics Aggregation Functions
# =============================================================================

def aggregate_wait_statistics(patients: List[PatientSession]) -> Tuple[float, float]:
    """
    Calculate mean and maximum wait times from patient sessions.
    
    Args:
        patients: List of patient sessions.
        
    Returns:
        Tuple of (mean_wait_time, max_wait_time).
    """
    valid_wait_times = [
        p.wait_time for p in patients 
        if p.wait_time < EXTREME_WAIT_PENALTY and p.wait_time >= 0
    ]
    
    if valid_wait_times:
        mean_wait = sum(valid_wait_times) / len(valid_wait_times)
        max_wait = max(valid_wait_times)
    else:
        mean_wait = 0.0
        max_wait = 0.0
    
    # Include extreme waits in max calculation if present
    extreme_waits = [p.wait_time for p in patients if p.wait_time >= EXTREME_WAIT_PENALTY]
    if extreme_waits:
        max_wait = max(max_wait, max(extreme_waits))
    
    return mean_wait, max_wait


def aggregate_overrun_statistics(patients: List[PatientSession],
                                  shift_end: int) -> int:
    """
    Calculate total shift overrun from patient sessions.
    
    Args:
        patients: List of patient sessions.
        shift_end: Official shift end time.
        
    Returns:
        Maximum overrun in minutes.
    """
    shift_overrun = 0
    for patient in patients:
        if patient.session_end < float('inf'):
            overrun = calculate_overrun(patient.session_end, shift_end)
            if overrun > 0:
                shift_overrun = max(shift_overrun, overrun)
    return shift_overrun


# =============================================================================
# Bipartite Constraint Resolution Functions
# =============================================================================

def find_earliest_bipartite_slot(
    patient_arrival: int,
    machine_avail_time: int,
    nurses: List[NurseState],
    safety_limit: int = MAX_TIME_SAFETY_LIMIT
) -> Optional[int]:
    """
    Find the earliest time when both a machine and a nurse are available.
    
    This implements the bipartite constraint: patient needs BOTH resources.
    
    Args:
        patient_arrival: Minute when patient arrives.
        machine_avail_time: Minute when machine is available.
        nurses: List of nurse states to check.
        safety_limit: Maximum time to search before giving up.
        
    Returns:
        Earliest time when both resources are available, or None if impossible.
    """
    # Patient cannot start before arrival or machine availability
    current_time = max(machine_avail_time, patient_arrival)
    
    iterations = 0
    while current_time < safety_limit:
        # Check if any nurse is available at current_time
        earliest_nurse = find_earliest_nurse_availability(nurses, current_time)
        
        if earliest_nurse <= current_time:
            # Found an available nurse
            return current_time
        
        # Advance to next nurse availability
        current_time = earliest_nurse
        iterations += 1
        
        if iterations > 10000:  # Additional safety against infinite loops
            return None
    
    return None


def mark_nurse_busy(nurse: NurseState, busy_until: int) -> None:
    """
    Mark a nurse as busy until a specific time.
    
    Args:
        nurse: The nurse state to update.
        busy_until: Time when the nurse will be free.
    """
    nurse.busy_until = busy_until


def mark_machine_busy(machine: MachineState, busy_until: int, 
                      patient_id: Optional[int] = None) -> None:
    """
    Mark a machine as busy until a specific time.
    
    Args:
        machine: The machine state to update.
        busy_until: Time when the machine will be free.
        patient_id: Optional patient ID to associate with this booking.
    """
    machine.busy_until = busy_until
    if patient_id is not None:
        machine.assigned_patient_id = patient_id
