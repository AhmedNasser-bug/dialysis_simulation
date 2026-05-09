import os
from src.config import SimulationConfig
from src.batcher.batcher import MonteCarloBatcher
from src.visualizer.visualizer import Visualizer


def main(output_dir: str = "outputs", n_iterations: int = 10) -> None:
    os.makedirs(output_dir, exist_ok=True)

    config = SimulationConfig()
    batcher = MonteCarloBatcher(
        config=config,
        strategy_ids=["FIFO", "FIXED"],
        n_iterations=n_iterations,
        global_seed=42,
    )
    dataset = batcher.run()

    viz = Visualizer()
    plots = {
        "wait_distribution.png": viz.plot_wait_distribution(dataset),
        "utilization.png": viz.plot_utilization(dataset),
        "paired_difference.png": viz.plot_paired_difference(dataset),
    }
    for filename, fig in plots.items():
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        fig.clf()


if __name__ == "__main__":
    main()
