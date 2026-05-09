from __future__ import annotations

from typing import Dict, List, Optional

from src.models import ShiftScenario, ShiftStatistics
from src.strategies.base_strategy import SchedulingStrategy
from src.strategies.utils import (
    MachineState,
    NurseState,
    PatientSession,
    MAX_TIME_SAFETY_LIMIT,
    EXTREME_WAIT_PENALTY,
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


class FixedStrategy(SchedulingStrategy):
    """
    Fixed Assignment Strategy: Patient i must wait for Machine i.

    In this strategy:
    - Each patient is assigned to a specific machine by index (patient 0 → machine 0, etc.)
    - Patient waits for their designated machine AND any available nurse
    - If the designated machine is defective, the patient experiences extreme wait times
    - Bipartite constraint: session starts only when both machine AND nurse are available
    - Nurse is released after setup_time; machine is locked for setup + session + cooldown
    """

    @property
    def name(self) -> str:
        return "Fixed Assignment"

    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        # Initialize machine states (only non-defective machines)
        machines: Dict[int, MachineState] = {}
        for mid, ready_time in scenario.machine_ready_times.items():
            is_defective = mid in scenario.defective_machine_ids
            if not is_defective:
                machines[mid] = MachineState(
                    id=mid,
                    ready_time=ready_time,
                )

        # Initialize nurse states
        nurses: List[NurseState] = initialize_nurse_states(scenario.nurse_count)

        # Create patient sessions sorted by ID (fixed assignment order)
        patients: List[PatientSession] = []
        for p in sorted(scenario.patient_arrivals, key=lambda x: x["id"]):
            patients.append(PatientSession(
                patient_id=p["id"],
                arrival_time=p["arrival_min"],
                setup_duration=p["setup_min"]
            ))

        # Process each patient in order of their ID (fixed assignment)
        for patient in patients:
            # Determine assigned machine (patient i → machine i)
            assigned_machine_id = patient.patient_id

            if assigned_machine_id not in machines:
                # Machine doesn't exist or is defective - extreme wait
                patient.wait_time = EXTREME_WAIT_PENALTY
                patient.session_start = int(EXTREME_WAIT_PENALTY)
                patient.session_end = int(EXTREME_WAIT_PENALTY)
                continue

            machine = machines[assigned_machine_id]

            # Find earliest time when both machine and a nurse are available
            earliest_start = self._find_earliest_start_fixed(patient, machine, nurses)

            if earliest_start is None:
                # No nurse available ever (shouldn't happen with valid config)
                patient.wait_time = EXTREME_WAIT_PENALTY
                patient.session_start = int(EXTREME_WAIT_PENALTY)
                patient.session_end = int(EXTREME_WAIT_PENALTY)
                continue

            patient.session_start = earliest_start
            patient.wait_time = calculate_wait_time(patient.arrival_time, earliest_start)
            patient.session_end = calculate_session_end(
                earliest_start, 
                patient.setup_duration,
                scenario.session_duration_minutes
            )

            # Update machine state: locked for setup + session + cooldown
            machine.busy_until = calculate_machine_busy_until(
                patient.session_end,
                scenario.machine_cooldown_minutes
            )
            machine.assigned_patient_id = patient.patient_id

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

    def _find_earliest_start_fixed(
        self,
        patient: PatientSession,
        machine: MachineState,
        nurses: List[NurseState]
    ) -> Optional[int]:
        """Find earliest time when machine and a nurse are both available."""
        # Machine must be ready and not busy
        machine_available_from = max(machine.ready_time, machine.busy_until)
        
        # Patient cannot start before arrival
        current_time = max(machine_available_from, patient.arrival_time)

        # Search for an available nurse slot
        while True:
            # Find earliest nurse availability at or after current_time
            earliest_nurse = float('inf')
            for nurse in nurses:
                if nurse.busy_until <= current_time:
                    # Nurse is available now
                    return current_time
                else:
                    earliest_nurse = min(earliest_nurse, nurse.busy_until)

            # No nurse available now, advance to next nurse free time
            if earliest_nurse == float('inf'):
                return None  # Should not happen

            current_time = earliest_nurse

            # Safety limit
            if current_time > MAX_TIME_SAFETY_LIMIT:
                return None

    def _get_available_nurse(
        self, nurses: List[NurseState], at_time: int
    ) -> Optional[NurseState]:
        """Get a nurse that is available at the given time."""
        for nurse in nurses:
            if nurse.busy_until <= at_time:
                return nurse
        return None