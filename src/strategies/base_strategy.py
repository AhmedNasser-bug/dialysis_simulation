from abc import ABC, abstractmethod

from src.models import ShiftScenario, ShiftStatistics


class SchedulingStrategy(ABC):
    """
    Abstract base class for dialysis scheduling strategies.

    All concrete strategies must implement the process_shift method to produce
    ShiftStatistics from a given ShiftScenario.
    """

    @abstractmethod
    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        """
        Process a single shift scenario and return performance statistics.

        Args:
            scenario: Immutable snapshot of shift initial conditions.

        Returns:
            ShiftStatistics containing all performance metrics for this shift.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the human-readable name of this strategy."""
        pass