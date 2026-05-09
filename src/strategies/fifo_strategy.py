from typing import List

from src.models import ShiftScenario, ShiftStatistics
from src.strategies.base_strategy import SchedulingStrategy
from src.strategies.utils import (
    MachineState, NurseState, PatientSession,
    initialize_machine_states, initialize_nurse_states,
    find_earliest_available_nurse, calculate_wait_time,
    aggregate_shift_statistics, EXTREME_WAIT_PENALTY
)

class FIFOStrategy(SchedulingStrategy):
    @property
    def name(self) -> str:
        return "FIFO"

    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        machines = initialize_machine_states(scenario)
        nurses = initialize_nurse_states(scenario.nurse_count)
        
        # Sort patients by arrival time for FIFO
        arrivals = sorted(scenario.patient_arrivals, key=lambda x: (x["arrival_min"], x["id"]))
        
        patients: List[PatientSession] = []
        
        for arr in arrivals:
            pid = arr["id"]
            arrival = arr["arrival_min"]
            setup_dur = arr["setup_min"]
            session_dur = arr["session_min"]
            
            # Find earliest available machine
            free_machines = sorted(machines.values(), key=lambda m: max(arrival, m.busy_until))
            if not free_machines:
                # No machines at all (e.g., all defective)
                session = PatientSession(
                    patient_id=pid,
                    arrival_time=arrival,
                    setup_duration=setup_dur,
                    wait_time=0.0,
                    session_end=0,
                    failed_to_serve=True
                )
                patients.append(session)
                continue
                
            machine = free_machines[0]
            machine_reserved = max(arrival, machine.busy_until)
            
            # Now patient sits in chair waiting for nurse
            nurse = find_earliest_available_nurse(nurses, machine_reserved)
                
            setup_start = max(machine_reserved, nurse.busy_until)
            setup_end = setup_start + setup_dur
            dialysis_end = setup_end + session_dur
            machine_free = dialysis_end + scenario.machine_cooldown_minutes
            
            # Update utilization and tracking
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
            session = PatientSession(
                patient_id=pid,
                arrival_time=arrival,
                setup_duration=setup_dur,
                machine_reserved_minute=machine_reserved,
                setup_start_time=setup_start,
                session_start=setup_end,
                session_end=dialysis_end,
                wait_time=wait_time
            )
            patients.append(session)
            
        return aggregate_shift_statistics(self.name, scenario, patients, machines, nurses)