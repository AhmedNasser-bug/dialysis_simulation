"""
Module 3: Monte Carlo Batcher

Orchestrates paired execution of strategies against identical scenarios
to enable statistically valid paired-difference testing.
"""
from __future__ import annotations

from typing import List, Dict
from src.config import SimulationConfig
from src.models import ShiftScenario, ShiftStatistics
from src.scenario_generator import generate_shift_scenario
from src.strategies.base_strategy import SchedulingStrategy


class MonteCarloBatcher:
    """
    Executes Monte Carlo simulation with paired-difference testing design.
    
    For each iteration:
    1. Generate ONE scenario from seed
    2. Run ALL strategies against that SAME scenario
    3. Collect ShiftStatistics from each run
    
    This ensures any variance in outcomes is attributable to strategy
    differences, not stochastic scenario variation.
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        strategies: List[SchedulingStrategy],
        n_iterations: int,
        global_seed: int = 0,
    ) -> None:
        """
        Initialize the Monte Carlo batcher.
        
        Args:
            config: Simulation configuration parameters.
            strategies: List of strategy instances to compare.
            n_iterations: Number of Monte Carlo iterations to run.
            global_seed: Base seed for reproducibility.
        """
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1")
        if len(strategies) < 1:
            raise ValueError("At least one strategy must be provided")
            
        self._config = config
        self._strategies = strategies
        self._n_iterations = n_iterations
        self._global_seed = global_seed
    
    def run(self) -> List[ShiftStatistics]:
        """
        Execute the Monte Carlo simulation.
        
        Returns:
            List of ShiftStatistics objects (n_iterations * n_strategies total).
            Each object conforms to Schema 'S'.
        """
        all_results: List[ShiftStatistics] = []
        
        for i in range(self._n_iterations):
            # Generate scenario with deterministic seed
            seed = self._global_seed + i
            scenario: ShiftScenario = generate_shift_scenario(
                config=self._config,
                seed=seed
            )
            
            # Run each strategy against the SAME scenario (paired design)
            for strategy in self._strategies:
                stats: ShiftStatistics = strategy.process_shift(scenario)
                all_results.append(stats)
        
        return all_results


def _resolve_strategies(strategy_ids: List[str]) -> List[SchedulingStrategy]:
    """
    Resolve strategy names to strategy instances.
    
    Args:
        strategy_ids: List of strategy names (e.g., ["FIFO", "FIXED"]).
        
    Returns:
        List of instantiated strategy objects.
    """
    from src.strategies.fifo_strategy import FIFOStrategy
    from src.strategies.fixed_strategy import FixedStrategy
    
    registry: Dict[str, SchedulingStrategy] = {
        "FIFO": FIFOStrategy(),
        "FIXED": FixedStrategy(),
    }
    
    resolved: List[SchedulingStrategy] = []
    for sid in strategy_ids:
        if sid not in registry:
            raise ValueError(f"Unknown strategy: {sid}. Available: {list(registry.keys())}")
        resolved.append(registry[sid])
    
    return resolved