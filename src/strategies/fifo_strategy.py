from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.models import ShiftScenario, ShiftStatistics
from src.strategies.base_strategy import SchedulingStrategy


@dataclass
class _MachineState:
    """Internal state tracker for a machine."""
    id: int
    ready_time: int
    busy_until: int = 0  # Machine locked until this time (setup + session + cooldown)


@dataclass
class _NurseState:
    """Internal state tracker for a nurse."""
    id: int
    busy_until: int = 0  # Nurse busy only during setup time


@dataclass
class _PatientSession:
    """Tracks a patient's session timing."""
    patient_id: int
    arrival_time: int
    setup_duration: int
    session_start: int = 0
    session_end: int = 0
    wait_time: float = 0.0


class FIFOStrategy(SchedulingStrategy):
    """
    First-In-First-Out Strategy: Patients take first globally available machine-nurse pair.

    In this strategy:
    - Patients are processed in order of arrival time (FIFO queue)
    - Each patient takes the first available machine AND nurse pair
    - Bipartite constraint: session starts only when both machine AND nurse are available
    - Nurse is released after setup_time; machine is locked for setup + session + cooldown
    """

    SESSION_DURATION = 240
    COOLDOWN_DURATION = 60
    SHIFT_END = 300

    @property
    def name(self) -> str:
        return "FIFO"

    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        # Initialize machine states (excluding defective machines from availability)
        machines: Dict[int, _MachineState] = {}
        for mid, ready_time in scenario.machine_ready_times.items():
            if mid not in scenario.defective_machine_ids:
                machines[mid] = _MachineState(id=mid, ready_time=ready_time)

        # Initialize nurse states
        nurses: List[_NurseState] = [
            _NurseState(id=i) for i in range(scenario.nurse_count)
        ]

        # Create patient sessions sorted by arrival time (FIFO order)
        patients: List[_PatientSession] = []
        for p in sorted(scenario.patient_arrivals, key=lambda x: x["arrival_min"]):
            patients.append(_PatientSession(
                patient_id=p["id"],
                arrival_time=p["arrival_min"],
                setup_duration=p["setup_min"]
            ))

        max_time = 0

        # Process each patient in FIFO order
        for patient in patients:
            # Find earliest time when any machine and any nurse are both available
            earliest_start, assigned_machine = self._find_earliest_start_fifo(
                patient, machines, nurses
            )

            if earliest_start is None or assigned_machine is None:
                # No resources available - extreme wait (shouldn't happen with valid config)
                patient.wait_time = float('inf')
                patient.session_start = float('inf')
                patient.session_end = float('inf')
                continue

            patient.session_start = earliest_start
            patient.wait_time = max(0, earliest_start - patient.arrival_time)
            patient.session_end = earliest_start + patient.setup_duration + self.SESSION_DURATION

            # Update machine state: locked for setup + session + cooldown
            machines[assigned_machine].busy_until = (
                patient.session_end + self.COOLDOWN_DURATION
            )

            # Update nurse state: only busy during setup
            nurse = self._get_available_nurse(nurses, earliest_start)
            if nurse is not None:
                nurse.busy_until = earliest_start + patient.setup_duration

            # Track max time for utilization calculation
            max_time = max(max_time, patient.session_end)

        # Calculate statistics
        wait_times = [p.wait_time for p in patients if p.wait_time < float('inf')]
        if wait_times:
            mean_wait = sum(wait_times) / len(wait_times)
            max_wait = max(wait_times)
        else:
            mean_wait = 0.0
            max_wait = 0.0

        # Calculate overrun
        shift_overrun = 0
        for patient in patients:
            if patient.session_end < float('inf'):
                overrun = int(patient.session_end - self.SHIFT_END)
                if overrun > 0:
                    shift_overrun = max(shift_overrun, overrun)

        # Calculate utilizations (as percentages)
        nurse_util, machine_util = self._calculate_utilization(
            patients, machines, nurses, scenario.nurse_count, max_time
        )

        return ShiftStatistics(
            strategy_name=self.name,
            total_patients_processed=len(patients),
            mean_wait_time_minutes=mean_wait,
            max_wait_time_minutes=max_wait,
            nurse_utilization_percent=nurse_util * 100,
            machine_utilization_percent=machine_util * 100,
            shift_overrun_minutes=shift_overrun
        )

    def _find_earliest_start_fifo(
        self,
        patient: _PatientSession,
        machines: Dict[int, _MachineState],
        nurses: List[_NurseState]
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Find earliest time when any machine and any nurse are both available.
        Returns (start_time, machine_id) or (None, None) if no slot found.
        """
        if not machines:
            return None, None

        # Patient cannot start before arrival
        earliest_possible = patient.arrival_time

        # Get all machine availability times >= earliest_possible
        machine_avails: List[Tuple[int, int]] = []  # (available_time, machine_id)
        for mid, m in machines.items():
            avail = max(m.ready_time, m.busy_until, earliest_possible)
            machine_avails.append((avail, mid))

        # Sort by availability time
        machine_avails.sort(key=lambda x: x[0])

        # Try to find a matching nurse slot
        for machine_avail_time, machine_id in machine_avails:
            # Check if any nurse is available at this time
            current_time = machine_avail_time

            while True:
                # Find earliest nurse availability at or after current_time
                earliest_nurse = float('inf')
                for nurse in nurses:
                    if nurse.busy_until <= current_time:
                        # Found an available nurse
                        return current_time, machine_id
                    else:
                        earliest_nurse = min(earliest_nurse, nurse.busy_until)

                # No nurse available now, advance to next nurse free time
                if earliest_nurse == float('inf'):
                    break  # Try next machine

                current_time = earliest_nurse

                # Safety limit
                if current_time > 1000000:
                    break

        return None, None

    def _get_available_nurse(
        self, nurses: List[_NurseState], at_time: int
    ) -> Optional[_NurseState]:
        """Get a nurse that is available at the given time."""
        for nurse in nurses:
            if nurse.busy_until <= at_time:
                return nurse
        return None

    def _calculate_utilization(
        self,
        patients: List[_PatientSession],
        machines: Dict[int, _MachineState],
        nurses: List[_NurseState],
        nurse_count: int,
        max_time: int
    ) -> Tuple[float, float]:
        """Calculate nurse and machine utilization."""
        if max_time <= 0 or max_time == float('inf'):
            return 0.0, 0.0

        # Nurse utilization: total setup time / (nurse_count * max_time)
        total_setup_time = sum(
            p.setup_duration for p in patients if p.session_start < float('inf')
        )
        nurse_util = total_setup_time / (nurse_count * max_time) if max_time > 0 else 0.0

        # Machine utilization: total session time / (num_machines * max_time)
        total_session_time = sum(
            self.SESSION_DURATION for p in patients
            if p.session_start < float('inf')
        )
        num_machines = len(machines)
        machine_util = (
            total_session_time / (num_machines * max_time)
            if max_time > 0 and num_machines > 0
            else 0.0
        )

        return min(1.0, nurse_util), min(1.0, machine_util)