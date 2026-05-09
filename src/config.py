from __future__ import annotations

from dataclasses import dataclass, field, replace
import random
from typing import Callable


@dataclass(frozen=True, slots=True)
class IntRange:
    """Inclusive integer range [low, high]."""

    low: int
    high: int

    def sample(self, rng: random.Random) -> int:
        if self.low > self.high:
            raise ValueError(f"Invalid range: low({self.low}) > high({self.high})")
        return rng.randint(self.low, self.high)


@dataclass(frozen=True, slots=True)
class UniformIntSampler:
    """Uniform integer sampler over an inclusive minute range."""

    minute_range: IntRange

    def __call__(self, rng: random.Random) -> int:
        return self.minute_range.sample(rng)


@dataclass(frozen=True, slots=True)
class UniformFloatSampler:
    """Uniform float sampler over [low, high]."""

    low: float
    high: float

    def __call__(self, rng: random.Random) -> float:
        if self.low > self.high:
            raise ValueError(f"Invalid range: low({self.low}) > high({self.high})")
        return rng.uniform(self.low, self.high)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """
    Central repository of DES parameters.

    Stochastic variables are represented as ranges or generator callables so the
    Monte Carlo batcher can override them per experiment (via dataclasses.replace).
    """

    # 1) Static temporal bounds
    shift_duration_minutes: int = 300
    machine_cooldown_minutes: int = 60

    # 2) Stochastic resource ranges
    session_duration_minutes_range: IntRange = IntRange(240, 360)
    total_machines: int = IntRange(15, 20)
    patient_volume: IntRange = IntRange(15, 20)
    nurse_count: IntRange = IntRange(2, 4)
    machine_ready_delay_minutes: IntRange = IntRange(0, 90)

    # 3) Stochastic event generators (callables accept rng and return minutes)
    arrival_minute_sampler: Callable[[random.Random], int] = field(
        default_factory=lambda: UniformIntSampler(IntRange(0, 60))
    )
    setup_duration_minutes_sampler: Callable[[random.Random], int] = field(
        default_factory=lambda: UniformIntSampler(IntRange(10, 20))
    )

    # 4) Probability metrics
    machine_defect_probability: float = 0.15

    def with_overrides(self, **kwargs) -> "SimulationConfig":
        """Convenience wrapper around dataclasses.replace."""
        return replace(self, **kwargs)

    def validate(self) -> None:
        if self.shift_duration_minutes <= 0:
            raise ValueError("shift_duration_minutes must be > 0")
        if self.session_duration_minutes_range.low <= 0:
            raise ValueError("session_duration_minutes_range.low must be > 0")
        if self.machine_cooldown_minutes < 0:
            raise ValueError("machine_cooldown_minutes must be >= 0")
        if self.machine_ready_delay_minutes.low < 0 or self.machine_ready_delay_minutes.high < 0:
            raise ValueError("machine_ready_delay_minutes bounds must be >= 0")
        if self.machine_ready_delay_minutes.low > self.machine_ready_delay_minutes.high:
            raise ValueError("machine_ready_delay_minutes must satisfy low <= high")
        if not (0.0 <= self.machine_defect_probability <= 1.0):
            raise ValueError("machine_defect_probability must be in [0, 1]")

    # Sampling helpers (kept small; scenario generation can call these)
    def sample_patient_count(self, rng: random.Random) -> int:
        return self.patient_volume.sample(rng)

    def sample_nurse_count(self, rng: random.Random) -> int:
        return self.nurse_count.sample(rng)

    def sample_arrival_minute(self, rng: random.Random) -> int:
        return int(self.arrival_minute_sampler(rng))

    def sample_setup_duration_minutes(self, rng: random.Random) -> int:
        return int(self.setup_duration_minutes_sampler(rng))

    def sample_machine_ready_delay_minutes(self, rng: random.Random) -> int:
        return self.machine_ready_delay_minutes.sample(rng)

    def sample_machine_is_defective(self, rng: random.Random) -> bool:
        return rng.random() < self.machine_defect_probability

    def sample_session_duration_minutes(self, rng: random.Random) -> int:
        return self.session_duration_minutes_range.sample(rng)
