"""
Statistics Aggregator Module

Decouples raw Monte Carlo outputs from visualization by providing
data transformation and statistical aggregation utilities.
"""

from dataclasses import asdict
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from src.models import ShiftStatistics


def create_dataframes(shift_statistics: List[ShiftStatistics]) -> Dict[str, pd.DataFrame]:
    """
    Transform raw List[ShiftStatistics] into structured DataFrames.
    
    Parameters
    ----------
    shift_statistics : List[ShiftStatistics]
        Raw output from MonteCarloBatcher.run()
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'raw_df': Complete DataFrame for scatter/boxplot rendering
        - 'summary_df': Aggregated DataFrame with confidence intervals and means
    """
    if not shift_statistics:
        raise ValueError("shift_statistics list cannot be empty")
    
    # Convert dataclasses to dictionaries then to DataFrame
    raw_df = pd.DataFrame([asdict(stat) for stat in shift_statistics])
    
    # Ensure proper column types
    raw_df = raw_df.astype({
        'strategy_name': 'category',
        'total_patients_processed': 'int32',
        'mean_wait_time_minutes': 'float64',
        'max_wait_time_minutes': 'float64',
        'nurse_utilization_percent': 'float64',
        'machine_utilization_percent': 'float64',
        'shift_overrun_minutes': 'int32'
    })
    
    # Calculate summary statistics per strategy
    summary_data = []
    
    for strategy_name in raw_df['strategy_name'].unique():
        strategy_data = raw_df[raw_df['strategy_name'] == strategy_name]
        
        n_iterations = len(strategy_data)
        
        # Mean of the Mean Wait Times
        mean_of_mean_wait = strategy_data['mean_wait_time_minutes'].mean()
        std_mean_wait = strategy_data['mean_wait_time_minutes'].std(ddof=1)
        ci_95_mean_wait = 1.96 * std_mean_wait / np.sqrt(n_iterations)
        
        # Mean of the Max Wait Times
        mean_of_max_wait = strategy_data['max_wait_time_minutes'].mean()
        std_max_wait = strategy_data['max_wait_time_minutes'].std(ddof=1)
        ci_95_max_wait = 1.96 * std_max_wait / np.sqrt(n_iterations)
        
        # Average utilization rates
        avg_nurse_util = strategy_data['nurse_utilization_percent'].mean()
        avg_machine_util = strategy_data['machine_utilization_percent'].mean()
        
        # Total accumulated shift overrun minutes
        total_overrun = strategy_data['shift_overrun_minutes'].sum()
        mean_overrun = strategy_data['shift_overrun_minutes'].mean()
        std_overrun = strategy_data['shift_overrun_minutes'].std(ddof=1)
        ci_95_overrun = 1.96 * std_overrun / np.sqrt(n_iterations) if n_iterations > 1 else 0.0
        
        summary_data.append({
            'strategy_name': strategy_name,
            'n_iterations': n_iterations,
            'mean_of_mean_wait_minutes': mean_of_mean_wait,
            'std_mean_wait_minutes': std_mean_wait,
            'ci_95_mean_wait_minutes': ci_95_mean_wait,
            'mean_of_max_wait_minutes': mean_of_max_wait,
            'std_max_wait_minutes': std_max_wait,
            'ci_95_max_wait_minutes': ci_95_max_wait,
            'avg_nurse_utilization_percent': avg_nurse_util,
            'avg_machine_utilization_percent': avg_machine_util,
            'total_shift_overrun_minutes': total_overrun,
            'mean_shift_overrun_minutes': mean_overrun,
            'std_shift_overrun_minutes': std_overrun,
            'ci_95_shift_overrun_minutes': ci_95_overrun
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Ensure proper column types for summary
    summary_df = summary_df.astype({
        'strategy_name': 'category',
        'n_iterations': 'int32'
    })
    
    return {
        'raw_df': raw_df,
        'summary_df': summary_df
    }


def calculate_paired_differences(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate paired differences between strategies for each iteration.
    
    This enables paired-difference t-testing by computing the delta
    between strategy pairs on a per-iteration (per-scenario) basis.
    
    Parameters
    ----------
    raw_df : pd.DataFrame
        DataFrame containing ShiftStatistics with iteration identifiers
    
    Returns
    -------
    pd.DataFrame
        DataFrame with paired difference statistics
    
    Raises
    ------
    ValueError
        If fewer than 2 strategies are present in the data
    """
    strategies = raw_df['strategy_name'].unique()
    
    if len(strategies) < 2:
        raise ValueError(
            f"Paired difference calculation requires at least 2 strategies; "
            f"found: {list(strategies)}"
        )
    
    # For paired differences, we need to ensure we're comparing the same iterations
    # Assuming the data is ordered such that iteration i for all strategies are consecutive
    # or we have an iteration_id column (which we should add in MonteCarloBatcher)
    
    # Pivot the data to have strategies as columns
    # This assumes we can pair by row order within each strategy group
    pivot_data = {}
    
    for strategy in strategies:
        strategy_data = raw_df[raw_df['strategy_name'] == strategy].reset_index(drop=True)
        pivot_data[strategy] = strategy_data
    
    # Calculate differences (FIFO - FIXED, or alphabetically first vs others)
    strategy_list = sorted(strategies)
    reference_strategy = strategy_list[0]
    
    differences = []
    
    for i in range(len(pivot_data[reference_strategy])):
        row_diff = {
            'iteration': i,
            'reference_strategy': reference_strategy
        }
        
        for strategy in strategy_list[1:]:
            ref_row = pivot_data[reference_strategy].iloc[i]
            comp_row = pivot_data[strategy].iloc[i]
            
            row_diff[f'delta_mean_wait_{strategy}_vs_{reference_strategy}'] = (
                comp_row['mean_wait_time_minutes'] - ref_row['mean_wait_time_minutes']
            )
            row_diff[f'delta_max_wait_{strategy}_vs_{reference_strategy}'] = (
                comp_row['max_wait_time_minutes'] - ref_row['max_wait_time_minutes']
            )
            row_diff[f'delta_overrun_{strategy}_vs_{reference_strategy}'] = (
                comp_row['shift_overrun_minutes'] - ref_row['shift_overrun_minutes']
            )
        
        differences.append(row_diff)
    
    diff_df = pd.DataFrame(differences)
    
    # Calculate summary statistics for the differences
    diff_summary = {
        'comparison': f"{reference_strategy} vs others",
        'n_pairs': len(diff_df)
    }
    
    for col in diff_df.columns:
        if col.startswith('delta_'):
            mean_diff = diff_df[col].mean()
            std_diff = diff_df[col].std(ddof=1) if len(diff_df) > 1 else 0.0
            ci_95_diff = 1.96 * std_diff / np.sqrt(len(diff_df)) if len(diff_df) > 1 else 0.0
            
            diff_summary[f'{col}_mean'] = mean_diff
            diff_summary[f'{col}_std'] = std_diff
            diff_summary[f'{col}_ci_95'] = ci_95_diff
    
    return diff_df, pd.DataFrame([diff_summary])


def generate_statistical_report(dataframes: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a human-readable statistical report from the aggregated data.
    
    Parameters
    ----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary containing 'raw_df' and 'summary_df'
    
    Returns
    -------
    str
        Formatted statistical report
    """
    summary_df = dataframes['summary_df']
    
    report_lines = [
        "=" * 70,
        "DIALYSIS SCHEDULING SIMULATION - STATISTICAL REPORT",
        "=" * 70,
        ""
    ]
    
    for _, row in summary_df.iterrows():
        report_lines.extend([
            f"Strategy: {row['strategy_name']}",
            f"  Iterations: {row['n_iterations']}",
            "",
            "  Wait Time Analysis:",
            f"    Mean of Mean Wait: {row['mean_of_mean_wait_minutes']:.2f} ± {row['ci_95_mean_wait_minutes']:.2f} min (95% CI)",
            f"    Mean of Max Wait:  {row['mean_of_max_wait_minutes']:.2f} ± {row['ci_95_max_wait_minutes']:.2f} min (95% CI)",
            "",
            "  Resource Utilization:",
            f"    Nurse Utilization:   {row['avg_nurse_utilization_percent']:.2f}%",
            f"    Machine Utilization: {row['avg_machine_utilization_percent']:.2f}%",
            "",
            "  Shift Overrun:",
            f"    Total Overrun:   {row['total_shift_overrun_minutes']} min",
            f"    Mean Overrun:    {row['mean_shift_overrun_minutes']:.2f} ± {row['ci_95_shift_overrun_minutes']:.2f} min (95% CI)",
            "",
            "-" * 70,
            ""
        ])
    
    return "\n".join(report_lines)
