"""
Main entry point for the Dialysis Unit Scheduling Simulation.

Wires together all four modules of the DES pipeline:
1. Scenario Creator (src/scenario_generator)
2. Strategy Processor (src/strategies/*)
3. Monte Carlo Batcher (src/monte_carlo_batcher)
4. Visualizer (src/visualizer)
"""
import os
from typing import List

from src.config import SimulationConfig
from src.monte_carlo_batcher import MonteCarloBatcher, _resolve_strategies
from src.visualizer import Visualizer
from src.strategies.base_strategy import SchedulingStrategy


def main(
    output_dir: str = "outputs",
    n_iterations: int = 100,
    strategy_ids: List[str] = None,
    global_seed: int = 42,
) -> None:
    """
    Execute the full simulation pipeline and generate visualization outputs.
    
    Args:
        output_dir: Directory to save generated plots.
        n_iterations: Number of Monte Carlo iterations to run.
        strategy_ids: List of strategy names to compare (default: ["FIFO", "FIXED"]).
        global_seed: Base seed for reproducibility.
    """
    if strategy_ids is None:
        strategy_ids = ["FIFO", "FIXED"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize configuration
    config = SimulationConfig()
    config.validate()
    
    print(f"Dialysis Unit Scheduling Simulation")
    print(f"====================================")
    print(f"Configuration:")
    print(f"  - Shift duration: {config.shift_duration_minutes} minutes")
    print(f"  - Session duration: {config.session_duration_minutes} minutes")
    print(f"  - Machine cooldown: {config.machine_cooldown_minutes} minutes")
    print(f"  - Patient volume range: [{config.patient_volume.low}, {config.patient_volume.high}]")
    print(f"  - Nurse count range: [{config.nurse_count.low}, {config.nurse_count.high}]")
    print(f"  - Machine defect probability: {config.machine_defect_probability}")
    print(f"  - Monte Carlo iterations: {n_iterations}")
    print(f"  - Strategies: {strategy_ids}")
    print(f"  - Global seed: {global_seed}")
    print()
    
    # Resolve strategies
    strategies: List[SchedulingStrategy] = _resolve_strategies(strategy_ids)
    
    # Initialize and run Monte Carlo batcher
    print("Running Monte Carlo simulation...")
    batcher = MonteCarloBatcher(
        config=config,
        strategies=strategies,
        n_iterations=n_iterations,
        global_seed=global_seed,
    )
    results = batcher.run()
    print(f"Completed {len(results)} simulation runs ({n_iterations} iterations × {len(strategies)} strategies)")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    viz = Visualizer()
    
    plots = {
        "wait_distribution.png": viz.plot_wait_distribution(results),
        "utilization.png": viz.plot_utilization(results),
        "overrun_histogram.png": viz.plot_overrun_histogram(results),
        "paired_difference.png": viz.plot_paired_difference(results),
    }
    
    for filename, fig in plots.items():
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        fig.clf()
        print(f"  Saved: {filepath}")
    
    print()
    print("Simulation complete!")
    print(f"Output files saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
