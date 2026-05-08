from __future__ import annotations

from dataclasses import dataclass
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
    {"id": 1, "arrival_min": 15, "setup_min": 14}
    """

    nurse_count: int

    machine_ready_times: Dict[int, int]
    """Machine id → minute when the machine becomes ready for use."""

    defective_machine_ids: List[int]
    """Machine ids that are permanently offline for the shift (defects at T=0)."""

    scenario_seed: int
    """Seed used to generate this scenario (supports reproducibility)."""
