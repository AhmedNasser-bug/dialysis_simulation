"""
Module 4: Analytical Visualizer

Transforms Monte Carlo simulation results into visual insights.
Operates strictly against Schema 'S' interface.
"""
from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt
import matplotlib.figure as mpfig
import pandas as pd
import seaborn as sns

from src.models import ShiftStatistics


class Visualizer:
    def plot_wait_distribution(
        self, 
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
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
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        
        df_melted = df.melt(
            id_vars=["strategy_name"],
            value_vars=["nurse_utilization_percent", "machine_utilization_percent"],
            var_name="resource_type",
            value_name="utilization_percent"
        )
        
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
        ax.set_ylabel("Utilization (fraction)")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Resource Type")
        
        plt.tight_layout()
        return fig
    
    def plot_overrun_histogram(
        self,
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
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
        ax.set_xlabel("Overrun Minutes")
        ax.set_ylabel("Frequency")
        ax.legend(title="Strategy")
        
        plt.tight_layout()
        return fig
    
    def plot_paired_difference(
        self,
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        strategies = df["strategy_name"].unique()
        
        if len(strategies) < 2:
            raise ValueError(
                f"Paired-difference plot requires at least 2 distinct strategies; "
                f"found: {list(strategies)}"
            )
        
        pivot_df = df.pivot_table(
            index=df.index // len(strategies),
            columns="strategy_name",
            values="mean_wait_time_minutes",
            aggfunc='first'
        )
        
        strat_names = sorted(pivot_df.columns)
        if len(strat_names) >= 2:
            diff_col = f"{strat_names[0]}_minus_{strat_names[1]}"
            pivot_df[diff_col] = pivot_df[strat_names[0]] - pivot_df[strat_names[1]]
            
            diffs = pivot_df[diff_col].dropna()
            mean_diff = diffs.mean()
            std_err = diffs.std() / (len(diffs) ** 0.5)
            ci_95 = 1.96 * std_err
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
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
    
    def plot_metric_over_iterations(
        self,
        results: List[ShiftStatistics],
        metric_column: str,
        title: str,
        ylabel: str
    ) -> mpfig.Figure:
        """
        Line plot showing a specific metric across all iterations for each strategy,
        with a moving average applied for smoother visual understanding.
        """
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        strategies = df["strategy_name"].unique()
        
        # Calculate iteration index
        df["iteration"] = df.index // len(strategies)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw data faintly in the background
        sns.lineplot(
            data=df,
            x="iteration",
            y=metric_column,
            hue="strategy_name",
            alpha=0.2,
            linewidth=1,
            legend=False,
            ax=ax
        )
        
        # Calculate rolling moving average (dynamic window size based on iteration count)
        total_iterations = len(df) // len(strategies)
        window_size = max(3, total_iterations // 20) # 5% rolling window
        
        df["smoothed_metric"] = df.groupby("strategy_name")[metric_column].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        
        # Plot smoothed data solidly
        sns.lineplot(
            data=df,
            x="iteration",
            y="smoothed_metric",
            hue="strategy_name",
            linewidth=2.5,
            ax=ax
        )
        
        ax.set_title(f"{title} (Smoothed Window: {window_size})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_single_wait_distribution(
        self,
        strategy_name: str,
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        """Plot wait time distribution for a single strategy."""
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        sns.histplot(data=df, x="mean_wait_time_minutes", kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f"Mean Wait Time - {strategy_name}")
        axes[0].set_xlabel("Mean Wait Time (min)")
        
        sns.histplot(data=df, x="max_wait_time_minutes", kde=True, ax=axes[1], color="salmon")
        axes[1].set_title(f"Max Wait Time - {strategy_name}")
        axes[1].set_xlabel("Max Wait Time (min)")
        
        plt.tight_layout()
        return fig

    def plot_single_utilization(
        self,
        strategy_name: str,
        results: List[ShiftStatistics]
    ) -> mpfig.Figure:
        """Plot resource utilization density for a single strategy using Violins."""
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        
        # Melt DataFrame for seaborn violin plot
        melted = pd.melt(
            df, 
            value_vars=["nurse_utilization_percent", "machine_utilization_percent"],
            var_name="Resource", 
            value_name="Utilization (%)"
        )
        melted["Utilization (%)"] *= 100
        melted["Resource"] = melted["Resource"].replace({
            "nurse_utilization_percent": "Nurse",
            "machine_utilization_percent": "Machine"
        })
        
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.violinplot(data=melted, x="Resource", y="Utilization (%)", ax=ax, palette="viridis", inner="quartile")
        ax.set_title(f"Resource Utilization Density - {strategy_name}")
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        return fig

    def plot_global_utilization_violin(self, results: List[ShiftStatistics]) -> mpfig.Figure:
        """Plot resource utilization density globally across strategies."""
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        melted = pd.melt(
            df,
            id_vars=["strategy_name"],
            value_vars=["nurse_utilization_percent", "machine_utilization_percent"],
            var_name="Resource",
            value_name="Utilization (%)"
        )
        melted["Utilization (%)"] *= 100
        melted["Resource"] = melted["Resource"].replace({
            "nurse_utilization_percent": "Nurse",
            "machine_utilization_percent": "Machine"
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=melted, x="Resource", y="Utilization (%)", hue="strategy_name", split=True if len(df["strategy_name"].unique())==2 else False, ax=ax, inner="quartile")
        ax.set_title("Resource Utilization Density Comparison")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        return fig

    def plot_cdf_wait_time(self, results: List[ShiftStatistics], strategy_name: str = None) -> mpfig.Figure:
        """Plot Cumulative Distribution Function for Mean Wait Time."""
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        if strategy_name:
            df = df[df["strategy_name"] == strategy_name]
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.ecdfplot(data=df, x="mean_wait_time_minutes", hue="strategy_name" if not strategy_name else None, ax=ax)
        
        title = f"CDF of Mean Wait Time - {strategy_name}" if strategy_name else "CDF of Mean Wait Time Comparison"
        ax.set_title(title)
        ax.set_xlabel("Mean Wait Time (min)")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_scatter_wait_vs_overrun(self, results: List[ShiftStatistics], strategy_name: str = None) -> mpfig.Figure:
        """Scatter plot showing trade-off between Wait Time and Shift Overrun."""
        df = pd.DataFrame([self._stats_to_dict(r) for r in results])
        if strategy_name:
            df = df[df["strategy_name"] == strategy_name]
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df, 
            x="mean_wait_time_minutes", 
            y="shift_overrun_minutes", 
            hue="strategy_name" if not strategy_name else None,
            alpha=0.7,
            ax=ax
        )
        
        title = f"Wait Time vs Shift Overrun - {strategy_name}" if strategy_name else "Wait Time vs Shift Overrun Comparison"
        ax.set_title(title)
        ax.set_xlabel("Mean Wait Time (min)")
        ax.set_ylabel("Shift Overrun (min)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def _stats_to_dict(stats: ShiftStatistics) -> dict:
        return {
            "strategy_name": stats.strategy_name,
            "total_patients_processed": stats.total_patients_processed,
            "mean_wait_time_minutes": stats.mean_wait_time_minutes,
            "max_wait_time_minutes": stats.max_wait_time_minutes,
            "nurse_utilization_percent": stats.nurse_utilization_percent,
            "machine_utilization_percent": stats.machine_utilization_percent,
            "shift_overrun_minutes": stats.shift_overrun_minutes,
            "failed_patients_count": stats.failed_patients_count,
        }