from typing import List

from src.models import ShiftScenario, ShiftStatistics
from src.strategies.base_strategy import SchedulingStrategy
from src.strategies.utils import (
    MachineState, NurseState, PatientSession,
    initialize_machine_states, initialize_nurse_states,
    find_earliest_available_nurse, calculate_wait_time,
    cap_session_to_shift, aggregate_shift_statistics, EXTREME_WAIT_PENALTY
)

class FixedStrategy(SchedulingStrategy):
    @property
    def name(self) -> str:
        return "FIXED"

    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        machines = initialize_machine_states(scenario)
        nurses = initialize_nurse_states(scenario.nurse_count)

        # In Fixed Assignment, Patient i goes to Machine i.
        # (Assuming machine IDs 1..N and Patient IDs 1..M)
        arrivals = sorted(scenario.patient_arrivals, key=lambda x: x["id"])

        patients: List[PatientSession] = []

        for arr in arrivals:
            pid = arr["id"]
            arrival = arr["arrival_min"]
            setup_dur = arr["setup_min"]
            session_dur = arr["session_min"]

            machine = machines.get(pid)
            if machine is None:
                # Machine is defective or doesn't exist
                patients.append(PatientSession(
                    patient_id=pid,
                    arrival_time=arrival,
                    setup_duration=setup_dur,
                    wait_time=0.0,
                    session_end=0,
                    failed_to_serve=True
                ))
                continue

            machine_reserved = max(arrival, machine.busy_until)
            nurse = find_earliest_available_nurse(nurses, machine_reserved)
            setup_start = max(machine_reserved, nurse.busy_until)
            setup_end = setup_start + setup_dur

            # Apply shift-wall cap
            actual_dur, is_truncated = cap_session_to_shift(setup_end, session_dur, scenario)

            # Patient cannot be served if session < minimum viable duration
            if actual_dur < scenario.min_session_duration_minutes:
                patients.append(PatientSession(
                    patient_id=pid,
                    arrival_time=arrival,
                    setup_duration=setup_dur,
                    wait_time=calculate_wait_time(arrival, setup_start),
                    session_end=0,
                    failed_to_serve=True
                ))
                continue

            dialysis_end = setup_end + int(actual_dur)
            machine_free = dialysis_end + scenario.machine_cooldown_minutes

            machine.busy_until = machine_free
            machine_busy_start = min(scenario.shift_end_minutes, machine_reserved)
            machine_busy_end = min(scenario.shift_end_minutes, dialysis_end)
            if machine_busy_end > machine_busy_start:
                machine.total_occupied_minutes += (machine_busy_end - machine_busy_start)

            nurse.busy_until = setup_end
            nurse.setups_completed += 1
            nurse_busy_start = min(scenario.shift_end_minutes, setup_start)
            nurse_busy_end = min(scenario.shift_end_minutes, setup_end)
            if nurse_busy_end > nurse_busy_start:
                nurse.total_occupied_minutes += (nurse_busy_end - nurse_busy_start)

            wait_time = calculate_wait_time(arrival, setup_start)
            patients.append(PatientSession(
                patient_id=pid,
                arrival_time=arrival,
                setup_duration=setup_dur,
                machine_reserved_minute=machine_reserved,
                setup_start_time=setup_start,
                session_start=setup_end,
                session_end=dialysis_end,
                actual_session_duration=actual_dur,
                prescribed_session_duration=float(session_dur),
                wait_time=wait_time
            ))

        return aggregate_shift_statistics(self.name, scenario, patients, machines, nurses)