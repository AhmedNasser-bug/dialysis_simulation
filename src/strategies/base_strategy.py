from abc import ABC, abstractmethod
from src.models import ShiftScenario, ShiftStatistics

class SchedulingStrategy(ABC):
    @abstractmethod
    def process_shift(self, scenario: ShiftScenario) -> ShiftStatistics:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass