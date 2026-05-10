from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True, slots=True)
class ShiftScenario:
    """
    Immutable, deterministic snapshot of a single shift's initial conditions.

    This object contains *only* the baseline constraints (arrivals, resources,
    defects, readiness). It does not include any time progression or scheduling
    outcomes; those are produced by the strategy processor (Module 2).
    """

    patient_arrivals: List[Dict[str, int]]
    """
    Each element is a row-like dict for patient i, e.g.:
    {"id": 1, "arrival_min": 15, "setup_min": 14, "session_min": 300}
    """

    nurse_count: int

    machine_ready_times: Dict[int, int]
    """Machine id -> minute when the machine becomes ready for use."""

    defective_machine_ids: List[int]
    """Machine ids that are permanently offline for the shift (defects at T=0)."""

    scenario_seed: int
    """Seed used to generate this scenario (supports reproducibility)."""

    machine_cooldown_minutes: int
    """Cooldown period after session completion in minutes."""

    shift_end_minutes: int
    """Official shift end time in minutes (hard wall -- sessions are cut here)."""

    min_session_duration_minutes: int
    """Minimum viable session length in minutes. Patients receiving less are marked failed."""


@dataclass(frozen=True, slots=True)
class ShiftStatistics:
    """
    Schema 'S' contract: Output statistics from processing a single shift.

    This dataclass captures all performance metrics required for comparing
    scheduling strategies under identical stochastic conditions.
    """

    strategy_name: str
    """Name of the scheduling strategy that produced these statistics."""

    total_patients_processed: int
    """Total number of patients processed in this shift."""

    mean_wait_time_minutes: float
    """Mean wait time (in minutes) across all patients."""

    max_wait_time_minutes: float
    """Maximum wait time (in minutes) experienced by any patient."""

    nurse_utilization_percent: float
    """Average nurse utilization as a percentage [0, 100]."""

    machine_utilization_percent: float
    """Average machine utilization as a percentage [0, 100]."""

    avg_session_time_minutes: float
    """Mean actual dialysis session duration (in minutes) across all served patients."""

    sessions_truncated_count: int
    """Number of patients whose session was cut short by the shift wall (but still served)."""

    failed_patients_count: int = 0
    """Number of patients who could not be served (session would be < min_session_duration)."""
