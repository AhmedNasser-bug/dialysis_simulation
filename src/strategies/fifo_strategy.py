from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.models import ShiftScenario, ShiftStatistics
from src.strategies.base_strategy import SchedulingStrategy
from src.strategies.utils import (
    MachineState,
    NurseState,
    PatientSession,
    MAX_TIME_SAFETY_LIMIT,
    initialize_machine_states,
    initialize_nurse_states,
    find_earliest_nurse_availability,
    calculate_session_end,
    calculate_machine_busy_until,
    calculate_wait_time,
    calculate_nurse_utilization,
    calculate_machine_utilization,
    aggregate_wait_statistics,
    aggregate_overrun_statistics,
)


class FIFOStrategy(SchedulingStrategy):
    """
    First-In-First-Out Strategy: Patients take first globally available machine-nurse pair.

    In this strategy:
    - Patients are processed in order of arrival time (FIFO queue)
    - Each patient takes the first available machine AND nurse pair
    - Bipartite constraint: session starts only when both machine AND nurse are available
    - Nurse is released after setup_time; machine is locked for setup + session + cooldown
    """

    @property
    def name(self) -> str:
        return "FIFO"

    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        # Initialize machine states (excluding defective machines from availability)
        machines: Dict[int, MachineState] = initialize_machine_states(scenario)

        # Initialize nurse states
        nurses: List[NurseState] = initialize_nurse_states(scenario.nurse_count)

        # Create patient sessions sorted by arrival time (FIFO order)
        patients: List[PatientSession] = []
        for p in sorted(scenario.patient_arrivals, key=lambda x: x["arrival_min"]):
            patients.append(PatientSession(
                patient_id=p["id"],
                arrival_time=p["arrival_min"],
                setup_duration=p["setup_min"]
            ))

        # Process each patient in FIFO order
        for patient in patients:
            # Find earliest time when any machine and any nurse are both available
            earliest_start, assigned_machine_id = self._find_earliest_start_fifo(
                patient, machines, nurses
            )

            if earliest_start is None or assigned_machine_id is None:
                # No resources available - extreme wait (shouldn't happen with valid config)
                patient.wait_time = float('inf')
                patient.session_start = int(MAX_TIME_SAFETY_LIMIT)
                patient.session_end = int(MAX_TIME_SAFETY_LIMIT)
                continue

            patient.session_start = earliest_start
            patient.wait_time = calculate_wait_time(patient.arrival_time, earliest_start)
            patient.session_end = calculate_session_end(
                earliest_start,
                patient.setup_duration,
                scenario.session_duration_minutes
            )

            # Update machine state: locked for setup + session + cooldown
            machines[assigned_machine_id].busy_until = calculate_machine_busy_until(
                patient.session_end,
                scenario.machine_cooldown_minutes
            )

            # Update nurse state: only busy during setup
            nurse = self._get_available_nurse(nurses, earliest_start)
            if nurse is not None:
                nurse.busy_until = earliest_start + patient.setup_duration

        # Calculate statistics using utility functions
        mean_wait, max_wait = aggregate_wait_statistics(patients)
        shift_overrun = aggregate_overrun_statistics(patients, scenario.shift_end_minutes)
        max_time = max(p.session_end for p in patients if p.session_end < float('inf')) if patients else 0

        nurse_util = calculate_nurse_utilization(patients, scenario.nurse_count, max_time)
        machine_util = calculate_machine_utilization(
            patients,
            len(machines),
            max_time,
            scenario.session_duration_minutes
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
        patient: PatientSession,
        machines: Dict[int, MachineState],
        nurses: List[NurseState]
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
                if current_time > MAX_TIME_SAFETY_LIMIT:
                    break

        return None, None

    def _get_available_nurse(
        self, nurses: List[NurseState], at_time: int
    ) -> Optional[NurseState]:
        """Get a nurse that is available at the given time."""
        for nurse in nurses:
            if nurse.busy_until <= at_time:
                return nurse
        return None