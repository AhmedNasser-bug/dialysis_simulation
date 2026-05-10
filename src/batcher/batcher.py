from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable

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

    def run(self, output_csv_path: Optional[str] = None) -> List[ShiftStatistics]:
        """Run Monte Carlo iterations; return flat list of ShiftStatistics.
        If output_csv_path is provided, writes results incrementally and returns an empty list."""
        results: List[ShiftStatistics] = []
        csv_file = None
        writer = None

        if output_csv_path:
            csv_file = open(output_csv_path, 'w', newline='', encoding='utf-8')
            field_names = [f.name for f in dataclasses.fields(ShiftStatistics)]
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()

        try:
            for i in range(self._n_iterations):
                seed = self._global_seed + i
                scenario = generate_shift_scenario(self._config, seed=seed)
                for strategy in self._strategies:
                    stats = strategy.process_shift(scenario)
                    if writer:
                        writer.writerow(dataclasses.asdict(stats))
                    else:
                        results.append(stats)
        finally:
            if csv_file:
                csv_file.close()

        return results

    def run_with_scenarios(
        self,
        output_csv_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[ShiftStatistics], EdgeCaseBundle]:
        """
        Run Monte Carlo iterations and automatically collect failed-patient
        shifts as named edge cases.

        If output_csv_path is provided, the main results list will be empty
        and stats will be written incrementally to the CSV.
        """
        results: List[ShiftStatistics] = []
        discovered: EdgeCaseBundle = {}
        
        csv_file = None
        writer = None
        if output_csv_path:
            csv_file = open(output_csv_path, 'w', newline='', encoding='utf-8')
            field_names = [f.name for f in dataclasses.fields(ShiftStatistics)]
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()

        extremes = {
            strat.name: {
                'best_stat': None, 'best_scenario': None, 'best_all_stats': None,
                'worst_stat': None, 'worst_scenario': None, 'worst_all_stats': None
            }
            for strat in self._strategies
        }

        try:
            for i in range(self._n_iterations):
                seed = self._global_seed + i
                scenario = generate_shift_scenario(self._config, seed=seed)

                iteration_stats = []
                for strategy in self._strategies:
                    stats = strategy.process_shift(scenario)
                    iteration_stats.append(stats)
                    
                    if writer:
                        writer.writerow(dataclasses.asdict(stats))
                    else:
                        results.append(stats)

                for stats in iteration_stats:
                    s_name = stats.strategy_name
                    curr_worst = extremes[s_name]['worst_stat']
                    
                    # Update worst (most failures, or highest wait)
                    if not curr_worst or \
                       (stats.failed_patients_count > curr_worst.failed_patients_count) or \
                       (stats.failed_patients_count == curr_worst.failed_patients_count and stats.mean_wait_time_minutes > curr_worst.mean_wait_time_minutes):
                        extremes[s_name]['worst_stat'] = stats
                        extremes[s_name]['worst_scenario'] = scenario
                        extremes[s_name]['worst_all_stats'] = iteration_stats
                        
                    curr_best = extremes[s_name]['best_stat']
                    
                    # Update best (fewest failures, or lowest wait)
                    if not curr_best or \
                       (stats.failed_patients_count < curr_best.failed_patients_count) or \
                       (stats.failed_patients_count == curr_best.failed_patients_count and stats.mean_wait_time_minutes < curr_best.mean_wait_time_minutes):
                        extremes[s_name]['best_stat'] = stats
                        extremes[s_name]['best_scenario'] = scenario
                        extremes[s_name]['best_all_stats'] = iteration_stats
                
                if progress_callback:
                    progress_callback(i + 1, self._n_iterations)
        finally:
            if csv_file:
                csv_file.close()

        # Build discovered from extremes
        for s_name, data in extremes.items():
            if data['best_scenario']:
                discovered[f"Auto: Best Case ({s_name})"] = (data['best_scenario'], data['best_all_stats'])
            if data['worst_scenario']:
                discovered[f"Auto: Worst Case ({s_name})"] = (data['worst_scenario'], data['worst_all_stats'])

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