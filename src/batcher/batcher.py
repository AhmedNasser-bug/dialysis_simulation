from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.config import SimulationConfig
from src.models import ShiftScenario, ShiftStatistics
from src.scenario.generator import generate_shift_scenario
from src.strategies.base_strategy import SchedulingStrategy
from src.strategies.fifo_strategy import FIFOStrategy
from src.strategies.fixed_strategy import FixedStrategy

_STRATEGY_REGISTRY: Dict[str, SchedulingStrategy] = {
    "FIFO": FIFOStrategy(),
    "FIXED": FixedStrategy(),
}

# Type alias: edge case name -> (scenario_snapshot, [per-strategy stats])
EdgeCaseBundle = Dict[str, Tuple[ShiftScenario, List[ShiftStatistics]]]


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
        """Run Monte Carlo iterations; return flat list of ShiftStatistics."""
        results: List[ShiftStatistics] = []
        for i in range(self._n_iterations):
            seed = self._global_seed + i
            scenario = generate_shift_scenario(self._config, seed=seed)
            for strategy in self._strategies:
                stats = strategy.process_shift(scenario)
                results.append(stats)
        return results

    def run_with_scenarios(self) -> Tuple[List[ShiftStatistics], EdgeCaseBundle]:
        """
        Run Monte Carlo iterations and automatically collect failed-patient
        shifts as named edge cases.

        Returns
        -------
        results : List[ShiftStatistics]
            Flat list (same as ``run()``).
        discovered_edge_cases : EdgeCaseBundle
            Dict of auto-captured edge cases keyed as
            ``"Auto: seed-{seed} ({strategy})"`` for any shift where
            ``failed_patients_count > 0``.  Each value is a
            ``(ShiftScenario, [ShiftStatistics])`` tuple so the full shift
            snapshot is preserved alongside the strategy results.
        """
        results: List[ShiftStatistics] = []
        discovered: EdgeCaseBundle = {}

        for i in range(self._n_iterations):
            seed = self._global_seed + i
            scenario = generate_shift_scenario(self._config, seed=seed)

            for strategy in self._strategies:
                stats = strategy.process_shift(scenario)
                results.append(stats)

                if stats.failed_patients_count > 0:
                    label = f"Auto: seed-{seed} ({strategy.name})"
                    # Accumulate: if the same scenario triggered failures for
                    # multiple strategies, group them under the same key.
                    if label not in discovered:
                        discovered[label] = (scenario, [])
                    discovered[label][1].append(stats)

        return results, discovered

    def edge_case_run(self) -> EdgeCaseBundle:
        """
        Run all predefined edge-case scenarios.

        Returns
        -------
        EdgeCaseBundle
            ``{name: (ShiftScenario, [ShiftStatistics])}`` for each
            predefined edge case.
        """
        from src.scenario.edge_cases import get_all_edge_cases
        edge_cases = get_all_edge_cases(self._config)

        bundle: EdgeCaseBundle = {}
        for name, scenario in edge_cases.items():
            case_results: List[ShiftStatistics] = []
            for strategy in self._strategies:
                stats = strategy.process_shift(scenario)
                case_results.append(stats)
            bundle[name] = (scenario, case_results)

        return bundle