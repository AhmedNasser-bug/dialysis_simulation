"""
Module 4: Analytical Visualizer

Transforms Monte Carlo simulation results into visual insights.
Operates strictly against Schema 'S' interface.
"""
from __future__ import annotations

from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.figure as mpfig
import pandas as pd
import seaborn as sns

from src.models import ShiftStatistics


class Visualizer:
    """
    Generates analytical plots from Monte Carlo simulation results.
    
    All plot methods return matplotlib Figure objects without calling plt.show(),
    allowing callers to save or display as needed.
    """
    
    def plot_wait_distribution(
        self, 
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        """
        Box plot of wait time metrics grouped by strategy.
        
        Shows distribution of mean_wait_time_minutes and max_wait_time_minutes
        across all Monte Carlo iterations for each strategy.
        
        Args:
            results: List of ShiftStatistics from MonteCarloBatcher.run().
            
        Returns:
            Matplotlib Figure containing the box plot.
        """
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Mean wait time boxplot
        sns.boxplot(
            data=df, 
            x="strategy_name", 
            y="mean_wait_time_minutes",
            ax=axes[0]
        )
        axes[0].set_title("Mean Wait Time Distribution")
        axes[0].set_xlabel("Strategy")
        axes[0].set_ylabel("Mean Wait Time (minutes)")
        axes[0].tick_params(axis='x', rotation=45)
        
        # Max wait time boxplot
        sns.boxplot(
            data=df,
            x="strategy_name",
            y="max_wait_time_minutes", 
            ax=axes[1]
        )
        axes[1].set_title("Maximum Wait Time Distribution")
        axes[1].set_xlabel("Strategy")
        axes[1].set_ylabel("Max Wait Time (minutes)")
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_utilization(
        self,
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        """
        Grouped bar chart comparing nurse vs machine utilization by strategy.
        
        Args:
            results: List of ShiftStatistics from MonteCarloBatcher.run().
            
        Returns:
            Matplotlib Figure containing the utilization comparison.
        """
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        
        # Melt for grouped bar chart
        df_melted = df.melt(
            id_vars=["strategy_name"],
            value_vars=["nurse_utilization_percent", "machine_utilization_percent"],
            var_name="resource_type",
            value_name="utilization_percent"
        )
        
        # Clean up labels
        df_melted["resource_type"] = df_melted["resource_type"].map({
            "nurse_utilization_percent": "Nurse",
            "machine_utilization_percent": "Machine"
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=df_melted,
            x="strategy_name",
            y="utilization_percent",
            hue="resource_type",
            ax=ax
        )
        
        ax.set_title("Resource Utilization by Strategy")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Utilization (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Resource Type")
        
        plt.tight_layout()
        return fig
    
    def plot_overrun_histogram(
        self,
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        """
        Histogram of shift overrun minutes by strategy.
        
        Shows the distribution of extra time required beyond the 300-minute
        shift to complete all patient sessions.
        
        Args:
            results: List of ShiftStatistics from MonteCarloBatcher.run().
            
        Returns:
            Matplotlib Figure containing the histogram.
        """
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = df["strategy_name"].unique()
        colors = sns.color_palette(n_colors=len(strategies))
        
        for idx, strategy in enumerate(sorted(strategies)):
            subset = df[df["strategy_name"] == strategy]["shift_overrun_minutes"]
            ax.hist(
                subset,
                bins=20,
                alpha=0.7,
                label=strategy,
                color=colors[idx],
                edgecolor='black'
            )
        
        ax.set_title("Shift Overrun Distribution")
        ax.set_xlabel("Overrun Minutes (beyond 300 min shift)")
        ax.set_ylabel("Frequency")
        ax.legend(title="Strategy")
        
        plt.tight_layout()
        return fig
    
    def plot_paired_difference(
        self,
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        """
        Paired-difference plot with 95% confidence intervals.
        
        Computes per-scenario deltas between strategies and displays
        mean difference with confidence intervals.
        
        Args:
            results: List of ShiftStatistics from MonteCarloBatcher.run().
            
        Returns:
            Matplotlib Figure containing the paired difference plot.
            
        Raises:
            ValueError: If fewer than 2 distinct strategies are present.
        """
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        strategies = df["strategy_name"].unique()
        
        if len(strategies) < 2:
            raise ValueError(
                f"Paired-difference plot requires at least 2 distinct strategies; "
                f"found: {list(strategies)}"
            )
        
        # For paired analysis, we need to pair by iteration
        # Assuming results are ordered: [iter0_stratA, iter0_stratB, iter1_stratA, ...]
        n_iterations = len(results) // len(strategies)
        
        # Pivot to get paired data
        pivot_df = df.pivot_table(
            index=df.index // len(strategies),
            columns="strategy_name",
            values="mean_wait_time_minutes",
            aggfunc='first'
        )
        
        # Calculate differences (FIFO - FIXED, or first two strategies)
        strat_names = sorted(pivot_df.columns)
        if len(strat_names) >= 2:
            diff_col = f"{strat_names[0]}_minus_{strat_names[1]}"
            pivot_df[diff_col] = pivot_df[strat_names[0]] - pivot_df[strat_names[1]]
            
            diffs = pivot_df[diff_col].dropna()
            mean_diff = diffs.mean()
            std_err = diffs.std() / (len(diffs) ** 0.5)
            ci_95 = 1.96 * std_err
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot mean difference with CI
            ax.errorbar(
                [diff_col],
                [mean_diff],
                yerr=[[ci_95], [ci_95]],
                fmt='o',
                capsize=10,
                markersize=12,
                color='steelblue'
            )
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No Difference')
            ax.set_title(f"Paired Difference: {strat_names[0]} vs {strat_names[1]}\n"
                        f"Mean Diff: {mean_diff:.2f} min [95% CI: ±{ci_95:.2f}]")
            ax.set_ylabel("Mean Wait Time Difference (minutes)")
            ax.set_xticks([0])
            ax.set_xticklabels([f"{strat_names[0]} - {strat_names[1]}"])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
        
        raise ValueError("Could not compute paired differences")
    
    @staticmethod
    def _stats_to_dict(stats: ShiftStatistics) -> dict:
        """Convert ShiftStatistics to dictionary for DataFrame conversion."""
        return {
            "strategy_name": stats.strategy_name,
            "total_patients_processed": stats.total_patients_processed,
            "mean_wait_time_minutes": stats.mean_wait_time_minutes,
            "max_wait_time_minutes": stats.max_wait_time_minutes,
            "nurse_utilization_percent": stats.nurse_utilization_percent,
            "machine_utilization_percent": stats.machine_utilization_percent,
            "shift_overrun_minutes": stats.shift_overrun_minutes,
        }