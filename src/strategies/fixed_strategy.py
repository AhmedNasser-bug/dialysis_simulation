from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.models import ShiftScenario, ShiftStatistics
from src.strategies.base_strategy import SchedulingStrategy


@dataclass
class _MachineState:
    """Internal state tracker for a machine."""
    id: int
    ready_time: int
    is_defective: bool
    busy_until: int = 0  # Machine locked until this time (setup + session + cooldown)
    assigned_patient_id: Optional[int] = None


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

    SESSION_DURATION = 240
    COOLDOWN_DURATION = 60
    SHIFT_END = 300

    @property
    def name(self) -> str:
        return "Fixed Assignment"

    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        # Initialize machine states
        machines: Dict[int, _MachineState] = {}
        for mid, ready_time in scenario.machine_ready_times.items():
            is_defective = mid in scenario.defective_machine_ids
            machines[mid] = _MachineState(
                id=mid,
                ready_time=ready_time,
                is_defective=is_defective
            )

        # Initialize nurse states
        nurses: List[_NurseState] = [
            _NurseState(id=i) for i in range(scenario.nurse_count)
        ]

        # Create patient sessions sorted by arrival time
        patients: List[_PatientSession] = []
        for p in sorted(scenario.patient_arrivals, key=lambda x: x["id"]):
            patients.append(_PatientSession(
                patient_id=p["id"],
                arrival_time=p["arrival_min"],
                setup_duration=p["setup_min"]
            ))

        # Track statistics
        total_nurse_busy_time = 0
        total_machine_busy_time = 0
        max_time = 0

        # Process each patient in order of their ID (fixed assignment)
        for patient in patients:
            # Determine assigned machine (patient i → machine i)
            assigned_machine_id = patient.patient_id

            if assigned_machine_id not in machines:
                # Machine doesn't exist - extreme wait
                patient.wait_time = float('inf')
                patient.session_start = float('inf')
                patient.session_end = float('inf')
                continue

            machine = machines[assigned_machine_id]

            if machine.is_defective:
                # Defective machine - extreme wait time (use large finite value for stats)
                # Patient can never be served
                patient.wait_time = 999999.0
                patient.session_start = 999999
                patient.session_end = 999999
                continue

            # Find earliest time when both machine and a nurse are available
            earliest_start = self._find_earliest_start_fixed(
                patient, machine, nurses
            )

            if earliest_start is None:
                # No nurse available ever (shouldn't happen with valid config)
                patient.wait_time = float('inf')
                continue

            patient.session_start = earliest_start
            patient.wait_time = max(0, earliest_start - patient.arrival_time)
            patient.session_end = earliest_start + patient.setup_duration + self.SESSION_DURATION

            # Update machine state: locked for setup + session + cooldown
            machine.busy_until = patient.session_end + self.COOLDOWN_DURATION
            machine.assigned_patient_id = patient.patient_id

            # Update nurse state: only busy during setup
            nurse = self._get_available_nurse(nurses, earliest_start)
            if nurse is not None:
                nurse.busy_until = earliest_start + patient.setup_duration

            # Track max time for utilization calculation
            max_time = max(max_time, patient.session_end)

        # Calculate statistics
        # Include all valid wait times (excluding infinite waits from defective machines)
        valid_wait_times = [p.wait_time for p in patients if p.wait_time < 999999.0 and p.wait_time >= 0]
        
        # For mean/max calculations, only include patients who were actually served
        if valid_wait_times:
            mean_wait = sum(valid_wait_times) / len(valid_wait_times)
            max_wait = max(valid_wait_times)
        else:
            mean_wait = 0.0
            max_wait = 0.0
        
        # If there are patients with extreme waits (defective machines), include them in max
        extreme_waits = [p.wait_time for p in patients if p.wait_time >= 999999.0]
        if extreme_waits:
            max_wait = max(max_wait, max(extreme_waits))

        # Calculate overrun
        shift_overrun = 0
        for patient in patients:
            if patient.session_end != 0 and patient.session_end < float('inf'):
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

    def _find_earliest_start_fixed(
        self,
        patient: _PatientSession,
        machine: _MachineState,
        nurses: List[_NurseState]
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
            if current_time > 1000000:
                return None

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
        total_setup_time = sum(p.setup_duration for p in patients if p.session_start < float('inf'))
        nurse_util = total_setup_time / (nurse_count * max_time) if max_time > 0 else 0.0

        # Machine utilization: total (session) time / (num_machines * max_time)
        total_session_time = sum(
            self.SESSION_DURATION for p in patients
            if p.session_start < float('inf')
        )
        num_machines = len(machines)
        machine_util = total_session_time / (num_machines * max_time) if max_time > 0 and num_machines > 0 else 0.0

        return min(1.0, nurse_util), min(1.0, machine_util)