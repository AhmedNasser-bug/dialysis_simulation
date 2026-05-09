from typing import List, Dict

from src.config import SimulationConfig
from src.scenario.generator import generate_shift_scenario
from src.models import ShiftStatistics
from src.strategies.base_strategy import SchedulingStrategy
from src.strategies.fifo_strategy import FIFOStrategy
from src.strategies.fixed_strategy import FixedStrategy

_STRATEGY_REGISTRY: Dict[str, SchedulingStrategy] = {
    "FIFO": FIFOStrategy(),
    "FIXED": FixedStrategy(),
}

def _resolve_strategies(strategy_ids: List[str]) -> List[SchedulingStrategy]:
    return [_STRATEGY_REGISTRY[sid] for sid in strategy_ids]

class MonteCarloBatcher:
    def __init__(
        self,
        config: SimulationConfig,
        strategy_ids: List[str],
        n_iterations: int,
        global_seed: int = 0,
    ) -> None:
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        self._config = config
        self._strategies = _resolve_strategies(strategy_ids)
        self._n_iterations = n_iterations
        self._global_seed = global_seed

    def run(self) -> List[ShiftStatistics]:
        results: List[ShiftStatistics] = []

        for i in range(self._n_iterations):
            seed = self._global_seed + i
            scenario = generate_shift_scenario(self._config, seed=seed)

            for strategy in self._strategies:
                stats = strategy.process_shift(scenario)
                results.append(stats)

        return results