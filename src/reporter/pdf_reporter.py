import os
import tempfile
from typing import List, Dict
import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt

from src.config import SimulationConfig
from src.models import ShiftStatistics
from src.visualizer.visualizer import Visualizer

class PDFReporter(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Setup colors
        self.custom_brand_color = (41, 128, 185) # Navy Blue
        self.custom_text_color = (50, 50, 50)
        self.custom_light_gray = (240, 240, 240)
        
    def header(self):
        if self.page_no() == 1:
            return # No header on cover page
        self.set_font("Arial", "B", 10)
        self.set_text_color(*self.custom_brand_color)
        self.cell(0, 10, "Dialysis Simulation Analytical Report", border=0, ln=1, align="R")
        self.set_draw_color(*self.custom_brand_color)
        self.line(10, 20, 200, 20)
        self.ln(5)
        self.set_text_color(*self.custom_text_color)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")
        self.set_text_color(*self.custom_text_color)

    def create_cover_page(self, title: str, subtitle: str):
        self.add_page()
        self.set_y(80)
        self.set_font("Arial", "B", 24)
        self.set_text_color(*self.custom_brand_color)
        self.cell(0, 15, title, ln=1, align="C")
        
        self.set_font("Arial", "I", 14)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, subtitle, ln=1, align="C")
        self.ln(20)
        
        from datetime import datetime
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align="C")
        self.set_text_color(*self.custom_text_color)
        
    def chapter_title(self, title: str):
        self.set_font("Arial", "B", 16)
        self.set_fill_color(*self.custom_light_gray)
        self.set_text_color(*self.custom_brand_color)
        self.cell(0, 12, f"  {title}", ln=1, fill=True)
        self.set_text_color(*self.custom_text_color)
        self.ln(4)
        
    def chapter_body(self, text: str):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 6, text)
        self.ln()

    def _save_fig_to_temp(self, fig) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        return path

    def add_plot(self, fig, w=180, center=True):
        img_path = self._save_fig_to_temp(fig)
        x = (210 - w) / 2 if center else None
        self.image(img_path, x=x, w=w)
        os.remove(img_path)
        self.ln(5)

    def draw_percentile_table(self, df: pd.DataFrame, strat: str = None):
        self.set_font("Arial", "B", 10)
        self.set_fill_color(*self.custom_brand_color)
        self.set_text_color(255, 255, 255)
        
        # Header
        cols = ["Metric", "Mean", "p50 (Median)", "p90", "p99"]
        col_w = [60, 30, 30, 30, 30]
        for i, c in enumerate(cols):
            self.cell(col_w[i], 8, c, border=1, fill=True, align="C")
        self.ln()
        
        self.set_text_color(*self.custom_text_color)
        self.set_font("Arial", "", 10)
        
        metrics = {
            "Mean Wait (min)": "mean_wait_time_minutes",
            "Max Wait (min)": "max_wait_time_minutes",
            "Overrun (min)": "shift_overrun_minutes"
        }
        
        fill = False
        for label, col in metrics.items():
            self.set_fill_color(*self.custom_light_gray)
            
            data = df[col] if not strat else df[df["strategy_name"] == strat][col]
            
            mean_val = f"{data.mean():.1f}"
            p50 = f"{np.percentile(data, 50):.1f}"
            p90 = f"{np.percentile(data, 90):.1f}"
            p99 = f"{np.percentile(data, 99):.1f}"
            
            self.cell(col_w[0], 8, label, border=1, fill=fill)
            self.cell(col_w[1], 8, mean_val, border=1, fill=fill, align="C")
            self.cell(col_w[2], 8, p50, border=1, fill=fill, align="C")
            self.cell(col_w[3], 8, p90, border=1, fill=fill, align="C")
            self.cell(col_w[4], 8, p99, border=1, fill=fill, align="C")
            self.ln()
            fill = not fill
        self.ln(5)

    def write_config_section(self, config: SimulationConfig, n_iterations: int):
        self.chapter_title("1. Run Configuration")
        text = f"""Monte Carlo Iterations: {n_iterations}
Patient Volume Range: {config.patient_volume.low} - {config.patient_volume.high}
Machine Count Range: {config.total_machines.low} - {config.total_machines.high}
Nurse Count Range: {config.nurse_count.low} - {config.nurse_count.high}
Session Duration (min): {config.session_duration_minutes_range.low} - {config.session_duration_minutes_range.high}
Defective Machine Chance: {config.machine_defect_probability * 100:.1f}%
Machine Ready Delay (min): {config.machine_ready_delay_minutes.low} - {config.machine_ready_delay_minutes.high}"""
        self.chapter_body(text)

    def write_edge_cases(self, edge_case_results: Dict[str, List[ShiftStatistics]], strategy_name: str = None):
        self.chapter_title("Edge Case Stress Tests")
        self.chapter_body("Performance under extreme, deterministic constraints.")
        
        for case_name, stats_list in edge_case_results.items():
            self.set_font("Arial", "B", 12)
            self.set_text_color(*self.custom_brand_color)
            self.cell(0, 8, f"Scenario: {case_name}", ln=1)
            self.set_text_color(*self.custom_text_color)
            
            for stat in stats_list:
                if strategy_name and stat.strategy_name != strategy_name:
                    continue
                
                # Check for critical failures
                failed = stat.failed_patients_count > 0
                if failed:
                    self.set_fill_color(255, 230, 230) # Light red
                    self.set_text_color(200, 0, 0)
                else:
                    self.set_fill_color(240, 255, 240) # Light green
                    self.set_text_color(0, 100, 0)
                    
                self.set_font("Arial", "B" if failed else "", 10)
                body = (f"Strategy: {stat.strategy_name} | "
                        f"Wait(Mean/Max): {stat.mean_wait_time_minutes:.1f} / {stat.max_wait_time_minutes:.1f} min | "
                        f"Failed: {stat.failed_patients_count} | Overrun: {stat.shift_overrun_minutes} min")
                self.cell(0, 8, body, ln=1, fill=True)
                
            self.set_text_color(*self.custom_text_color)
            self.ln(4)


def generate_individual_report(
    strategy_name: str,
    results: List[ShiftStatistics],
    edge_cases: Dict[str, List[ShiftStatistics]],
    config: SimulationConfig,
    n_iterations: int,
    output_path: str
):
    pdf = PDFReporter()
    viz = Visualizer()
    df = pd.DataFrame([viz._stats_to_dict(r) for r in results])
    
    # Filter results for this strategy
    strat_results = [r for r in results if r.strategy_name == strategy_name]
    if not strat_results:
        return
    
    pdf.create_cover_page(
        title=f"Strategy Profile: {strategy_name}", 
        subtitle="Isolated Performance and Resilience Analytics"
    )
    
    pdf.add_page()
    pdf.write_config_section(config, n_iterations)
    
    pdf.chapter_title("2. Statistical Summary")
    pdf.chapter_body("Percentile distributions across all Monte Carlo iterations.")
    pdf.draw_percentile_table(df, strategy_name)
    
    pdf.chapter_title("3. Distribution & Utilization")
    fig_wait = viz.plot_single_wait_distribution(strategy_name, strat_results)
    pdf.add_plot(fig_wait)
    
    fig_util = viz.plot_single_utilization(strategy_name, strat_results)
    pdf.add_plot(fig_util, w=120)
    
    pdf.add_page()
    pdf.chapter_title("4. Advanced Insights")
    fig_cdf = viz.plot_cdf_wait_time(results, strategy_name)
    pdf.add_plot(fig_cdf, w=150)
    
    fig_scatter = viz.plot_scatter_wait_vs_overrun(results, strategy_name)
    pdf.add_plot(fig_scatter, w=150)
    
    pdf.add_page()
    pdf.write_edge_cases(edge_cases, strategy_name)
    
    pdf.output(output_path)


def generate_global_comparison_report(
    results: List[ShiftStatistics],
    edge_cases: Dict[str, List[ShiftStatistics]],
    config: SimulationConfig,
    n_iterations: int,
    output_path: str
):
    pdf = PDFReporter()
    viz = Visualizer()
    df = pd.DataFrame([viz._stats_to_dict(r) for r in results])
    
    pdf.create_cover_page(
        title="Global Comparison Report", 
        subtitle="Monte Carlo Strategy Paired-Difference Analytics"
    )
    
    pdf.add_page()
    pdf.write_config_section(config, n_iterations)
    
    pdf.chapter_title("2. Global Statistical Summary")
    pdf.chapter_body("Percentile metrics aggregated across the full suite.")
    strategies = list(set(r.strategy_name for r in results))
    for strat in strategies:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, strat, ln=1)
        pdf.draw_percentile_table(df, strat)
        
    pdf.add_page()
    pdf.chapter_title("3. Comparative Analytics")
    try:
        fig_paired = viz.plot_paired_difference(results, "mean_wait_time_minutes", "Mean Wait Time")
        pdf.add_plot(fig_paired)
    except Exception:
        pass
        
    fig_dist = viz.plot_wait_distribution(results)
    pdf.add_plot(fig_dist)
    
    pdf.add_page()
    fig_util_global = viz.plot_global_utilization_violin(results)
    pdf.add_plot(fig_util_global)
    
    fig_cdf = viz.plot_cdf_wait_time(results)
    pdf.add_plot(fig_cdf)
    
    pdf.add_page()
    fig_scatter = viz.plot_scatter_wait_vs_overrun(results)
    pdf.add_plot(fig_scatter)
    
    pdf.write_edge_cases(edge_cases)
    
    pdf.output(output_path)
