import os
import tempfile
from datetime import datetime
from typing import List, Dict

import pandas as pd
import numpy as np
from fpdf import FPDF
import matplotlib.pyplot as plt

from src.config import SimulationConfig
from src.models import ShiftStatistics
from src.visualizer.visualizer import Visualizer

# ─── Design tokens ────────────────────────────────────────────────────────────
# Primary palette (HSL-tuned deep slate-blue + warm accent)
C_PRIMARY   = (22,  55,  99)   # Deep navy
C_ACCENT    = (37, 122, 168)   # Steel blue
C_GOLD      = (180, 135,  60)  # Warm ochre — used for accent rules/badges
C_LIGHT     = (245, 247, 250)  # Near-white panel fill
C_MID       = (210, 218, 228)  # Divider / border
C_TEXT      = (30,  35,  45)   # Near-black body text
C_MUTED     = (110, 120, 135)  # Secondary / caption text
C_WHITE     = (255, 255, 255)

# Row-stripe alternation
C_ROW_A = (245, 247, 250)
C_ROW_B = (255, 255, 255)

# Table header
C_TH_BG    = (22,  55,  99)
C_TH_TEXT  = (255, 255, 255)

# Status colours
C_PASS_BG   = (235, 248, 240)
C_PASS_TEXT = (22, 120,  60)
C_FAIL_BG   = (255, 235, 235)
C_FAIL_TEXT = (185,  30,  30)

# ─── Typography constants ─────────────────────────────────────────────────────
MARGIN_L = 16
MARGIN_R = 16
PAGE_W   = 210
BODY_W   = PAGE_W - MARGIN_L - MARGIN_R    # 178 mm


class PDFReporter(FPDF):
    """
    Academic-grade PDF reporter for the Dialysis Monte Carlo Simulation.
    Design language: statistical journal (Nature / NEJM inspired) — clean
    typographic hierarchy, ruled section breaks, structured parameter tables.
    """

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(MARGIN_L, 15, MARGIN_R)
        self._section_counter = 0

    # ─── Page furniture ───────────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return
        # Thin top rule
        self.set_draw_color(*C_MID)
        self.set_line_width(0.3)
        self.line(MARGIN_L, 12, PAGE_W - MARGIN_R, 12)
        # Running header text
        self.set_font("Arial", "I", 8)
        self.set_text_color(*C_MUTED)
        self.set_y(8)
        self.cell(0, 5, "Dialysis Simulation  |  Monte Carlo Analytical Report", align="L")
        self.cell(0, 5, datetime.now().strftime("%Y-%m-%d"), align="R")
        self.set_y(18)
        self.set_text_color(*C_TEXT)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*C_MID)
        self.set_line_width(0.3)
        self.line(MARGIN_L, self.get_y(), PAGE_W - MARGIN_R, self.get_y())
        self.set_font("Arial", "", 8)
        self.set_text_color(*C_MUTED)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(*C_TEXT)

    # ─── Cover page ───────────────────────────────────────────────────────────

    def create_cover_page(
        self,
        title: str,
        subtitle: str,
        n_iterations: int = None,
        n_strategies: int = None,
        config: "SimulationConfig" = None,
    ):
        self.add_page()

        # ── Full-width top colour band ──
        self.set_fill_color(*C_PRIMARY)
        self.rect(0, 0, PAGE_W, 68, "F")

        # ── Thin gold accent stripe below band ──
        self.set_fill_color(*C_GOLD)
        self.rect(0, 68, PAGE_W, 2, "F")

        # ── Report type label (small caps style) ──
        self.set_y(14)
        self.set_font("Arial", "B", 8)
        self.set_text_color(*C_GOLD)
        letter_spaced = "SIMULATION ANALYTICAL REPORT"
        self.cell(0, 6, letter_spaced, align="C")

        # ── Main title ──
        self.set_xy(MARGIN_L, 26)
        self.set_font("Arial", "B", 22)
        self.set_text_color(*C_WHITE)
        self.multi_cell(BODY_W, 11, title, align="C")

        # ── Subtitle ──
        self.set_x(MARGIN_L)
        self.set_font("Arial", "I", 11)
        self.set_text_color(180, 200, 220)
        self.multi_cell(BODY_W, 7, subtitle, align="C")

        # ── Metadata block (structured, below the band) ──
        self._cover_metadata_block(n_iterations, n_strategies)

        # ── Abstract / executive-summary label ──
        self.set_y(130)
        self._section_rule("Abstract", numbered=False)

        abstract = (
            "This report presents the results of a Monte Carlo discrete-event simulation "
            "study evaluating scheduling strategies for a haemodialysis unit.  "
            "Stochastic variability in patient volume, machine availability, session "
            "duration, and nursing resources is modelled across "
            + (f"{n_iterations:,}" if n_iterations else "multiple")
            + " independent runs per strategy.  "
            "Distributional summaries, tail-risk metrics (p90 / p99), and "
            "head-to-head strategy comparisons are provided to support operational "
            "decision-making."
        )
        self.set_x(MARGIN_L)
        self.set_font("Arial", "I", 9.5)
        self.set_text_color(*C_TEXT)
        self.multi_cell(BODY_W, 5.5, abstract)

        # ── Keywords line ──
        self.ln(4)
        self.set_font("Arial", "B", 8.5)
        self.set_text_color(*C_ACCENT)
        self.cell(22, 5, "Keywords:")
        self.set_font("Arial", "", 8.5)
        self.set_text_color(*C_MUTED)
        self.cell(0, 5, "Monte Carlo | Discrete-Event Simulation | Haemodialysis | Scheduling | Tail-Risk")

        self.set_text_color(*C_TEXT)

    def _cover_metadata_block(self, n_iterations, n_strategies):
        """Render the structured metadata card below the colour band."""
        block_top = 76
        self.set_y(block_top)

        # Outer light panel
        self.set_fill_color(*C_LIGHT)
        self.set_draw_color(*C_MID)
        self.set_line_width(0.3)
        self.rect(MARGIN_L, block_top, BODY_W, 44, "FD")

        # Left column label
        col_w = BODY_W / 3
        items = [
            ("Generated",        datetime.now().strftime("%d %B %Y  %H:%M")),
            ("Monte Carlo Runs", f"{n_iterations:,}" if n_iterations else "N/A"),
            ("Strategies Tested", str(n_strategies) if n_strategies else "N/A"),
            ("Method",           "Discrete-Event Simulation (DES)"),
            ("Unit",             "Haemodialysis Centre"),
            ("Report Type",      "Comparative Performance Analysis"),
        ]

        self.set_y(block_top + 5)
        for i, (label, value) in enumerate(items):
            col = i % 3
            row = i // 3
            x = MARGIN_L + col * col_w + 4
            y = block_top + 5 + row * 16

            self.set_xy(x, y)
            self.set_font("Arial", "B", 7.5)
            self.set_text_color(*C_MUTED)
            self.cell(col_w - 4, 5, label.upper())

            self.set_xy(x, y + 5.5)
            self.set_font("Arial", "B", 9.5)
            self.set_text_color(*C_PRIMARY)
            self.cell(col_w - 4, 5.5, value)

        self.set_y(block_top + 44 + 4)

    # ─── Section primitives ───────────────────────────────────────────────────

    def _section_rule(self, title: str, numbered: bool = True):
        """
        Render a chapter heading with a full-width bottom rule — the dominant
        visual hierarchy element in academic statistical reports.
        """
        if numbered:
            self._section_counter += 1
            label = f"{self._section_counter}.  {title}"
        else:
            label = title

        self.set_font("Arial", "B", 13)
        self.set_text_color(*C_PRIMARY)
        self.cell(0, 9, label, ln=1)

        # Gold accent line + lighter full rule
        self.set_fill_color(*C_GOLD)
        self.rect(MARGIN_L, self.get_y(), 30, 1.2, "F")
        self.set_draw_color(*C_MID)
        self.set_line_width(0.3)
        self.line(MARGIN_L + 30.5, self.get_y() + 0.6, PAGE_W - MARGIN_R, self.get_y() + 0.6)
        self.ln(6)
        self.set_text_color(*C_TEXT)

    def chapter_title(self, title: str):
        """Public alias kept for backward compatibility — delegates to _section_rule."""
        # Strip any leading numeric prefix if caller supplied it (e.g. "2. Foo")
        clean = title.lstrip("0123456789. ")
        self._section_rule(clean)

    def subsection_title(self, title: str):
        self.set_font("Arial", "B", 10.5)
        self.set_text_color(*C_ACCENT)
        self.cell(0, 7, title, ln=1)
        self.set_text_color(*C_TEXT)
        self.ln(1)

    def chapter_body(self, text: str):
        self.set_font("Arial", "", 10)
        self.set_text_color(*C_TEXT)
        self.multi_cell(BODY_W, 5.5, text)
        self.ln(3)

    def caption(self, text: str):
        self.set_font("Arial", "I", 8.5)
        self.set_text_color(*C_MUTED)
        self.multi_cell(BODY_W, 4.5, text)
        self.ln(3)
        self.set_text_color(*C_TEXT)

    # ─── Config section ───────────────────────────────────────────────────────

    def write_config_section(self, config: SimulationConfig, n_iterations: int):
        """
        Render the simulation configuration as a structured parameter table --
        grouped by category with labelled rows and units columns.
        """
        self._section_rule("Simulation Setup")

        self.chapter_body(
            "The parameters below fully specify the stochastic environment sampled "
            "during each Monte Carlo iteration.  All stochastic quantities are drawn "
            "independently per run using a seeded Mersenne Twister PRNG."
        )

        # ── Parameter groups ──────────────────────────────────────────────────
        groups = [
            {
                "label": "TEMPORAL BOUNDS",
                "rows": [
                    ("Shift Duration",              f"{config.shift_duration_minutes}",
                     "min", f"Hard wall -- {config.shift_duration_minutes // 60} h total shift"),
                    ("Nominal Session Duration",    f"{config.session_duration_minutes_range.low} - {config.session_duration_minutes_range.high}",
                     "min", "Prescribed per-patient session (~4 h target)"),
                    ("Min Viable Session",          f"{config.min_session_duration_minutes}",
                     "min", f"Sessions shorter than this are marked failed ({config.min_session_duration_minutes // 60} h)"),
                    ("Machine Cooldown",            f"{config.machine_cooldown_minutes}",
                     "min", "Mandatory inter-session machine rest period"),
                ],
            },
            {
                "label": "RESOURCE RANGES  (uniform draw per iteration)",
                "rows": [
                    ("Total Machines",           f"{config.total_machines.low} - {config.total_machines.high}",
                     "units", "Physical dialysis stations available"),
                    ("Patient Volume",           f"{config.patient_volume.low} - {config.patient_volume.high}",
                     "patients", "Demand sampled each shift"),
                    ("Nursing Staff",            f"{config.nurse_count.low} - {config.nurse_count.high}",
                     "nurses", "Concurrent supervision capacity"),
                    ("Session Duration",         f"{config.session_duration_minutes_range.low} - {config.session_duration_minutes_range.high}",
                     "min", "Per-patient dialysis time"),
                    ("Machine Ready Delay",      f"{config.machine_ready_delay_minutes.low} - {config.machine_ready_delay_minutes.high}",
                     "min", "Post-cooldown preparation lag"),
                ],
            },
            {
                "label": "STOCHASTIC EVENT GENERATORS",
                "rows": [
                    ("Patient Arrival Offset",   "Uniform[0, 60]",
                     "min", "Arrival jitter relative to shift start"),
                    ("Setup Duration",           "Uniform[10, 20]",
                     "min", "Machine preparation time per patient"),
                ],
            },
            {
                "label": "PROBABILITY METRICS",
                "rows": [
                    ("Machine Defect Probability", f"{config.machine_defect_probability * 100:.1f}%",
                     "--", "Independent Bernoulli draw per machine per session"),
                ],
            },
            {
                "label": "EXPERIMENT PARAMETERS",
                "rows": [
                    ("Monte Carlo Iterations",  f"{n_iterations:,}",
                     "runs", "Independent replications per strategy"),
                ],
            },
        ]

        col_widths = [52, 32, 16, BODY_W - 52 - 32 - 16]  # Parameter / Value / Unit / Note

        for group in groups:
            self._config_group_header(group["label"])
            self._config_table_header(col_widths)
            for i, row in enumerate(group["rows"]):
                self._config_table_row(row, col_widths, i)
            self.ln(4)

        self.ln(2)

    def _config_group_header(self, label: str):
        self.set_font("Arial", "B", 8)
        self.set_text_color(*C_GOLD)
        self.cell(0, 5.5, label, ln=1)
        self.set_text_color(*C_TEXT)

    def _config_table_header(self, col_widths: list):
        headers = ["Parameter", "Value", "Unit", "Description"]
        self.set_fill_color(*C_TH_BG)
        self.set_text_color(*C_TH_TEXT)
        self.set_font("Arial", "B", 8.5)
        for i, (h, w) in enumerate(zip(headers, col_widths)):
            self.cell(w, 7, h, border=1, fill=True, align="C" if i > 0 else "L")
        self.ln()

    def _config_table_row(self, row: tuple, col_widths: list, idx: int):
        param, value, unit, note = row
        fill_color = C_ROW_A if idx % 2 == 0 else C_ROW_B
        self.set_fill_color(*fill_color)
        self.set_text_color(*C_TEXT)

        # Parameter (left-aligned, slightly bold)
        self.set_font("Arial", "B", 8.5)
        self.cell(col_widths[0], 7, f"  {param}", border=1, fill=True)
        # Value (monospace feel — centred)
        self.set_font("Arial", "", 8.5)
        self.cell(col_widths[1], 7, value, border=1, fill=True, align="C")
        # Unit
        self.set_font("Arial", "I", 8)
        self.set_text_color(*C_MUTED)
        self.cell(col_widths[2], 7, unit, border=1, fill=True, align="C")
        # Description
        self.set_font("Arial", "", 8)
        self.set_text_color(*C_TEXT)
        self.cell(col_widths[3], 7, note, border=1, fill=True)
        self.ln()

    # | Statistical summary table |

    def draw_percentile_table(self, df: pd.DataFrame, strat: str = None):
        cols    = ["Metric", "Mean +/- SD", "p50", "p90", "p99"]
        col_w   = [58, 38, 26, 26, 26]

        # Header
        self.set_fill_color(*C_TH_BG)
        self.set_text_color(*C_TH_TEXT)
        self.set_font("Arial", "B", 8.5)
        for h, w in zip(cols, col_w):
            self.cell(w, 7, h, border=1, fill=True, align="C")
        self.ln()

        self.set_text_color(*C_TEXT)
        self.set_font("Arial", "", 8.5)

        metrics = {
            "Mean Wait Time (min)":       "mean_wait_time_minutes",
            "Max Wait Time (min)":        "max_wait_time_minutes",
            "Avg Session Time (min)":     "avg_session_time_minutes",
        }

        for i, (label, col) in enumerate(metrics.items()):
            fill_color = C_ROW_A if i % 2 == 0 else C_ROW_B
            self.set_fill_color(*fill_color)

            data = df[col] if strat is None else df[df["strategy_name"] == strat][col]
            mean_sd = f"{data.mean():.1f} +/- {data.std():.1f}"
            p50 = f"{np.percentile(data, 50):.1f}"
            p90 = f"{np.percentile(data, 90):.1f}"
            p99 = f"{np.percentile(data, 99):.1f}"

            self.set_font("Arial", "B", 8.5)
            self.cell(col_w[0], 7, f"  {label}", border=1, fill=True)
            self.set_font("Arial", "", 8.5)
            self.cell(col_w[1], 7, mean_sd, border=1, fill=True, align="C")
            self.cell(col_w[2], 7, p50,     border=1, fill=True, align="C")
            self.cell(col_w[3], 7, p90,     border=1, fill=True, align="C")
            self.cell(col_w[4], 7, p99,     border=1, fill=True, align="C")
            self.ln()

        self.ln(5)

    # ─── Edge case section ────────────────────────────────────────────────────

    def write_edge_cases(self, edge_case_results: Dict[str, List[ShiftStatistics]], strategy_name: str = None):
        self._section_rule("Edge-Case Stress Tests")
        self.chapter_body(
            "Performance under extreme, deterministic boundary conditions designed "
            "to expose failure modes not visible in the central Monte Carlo distribution."
        )

        for case_name, stats_list in edge_case_results.items():
            self.subsection_title(f"Scenario: {case_name}")

            for stat in stats_list:
                if strategy_name and stat.strategy_name != strategy_name:
                    continue

                failed = stat.failed_patients_count > 0
                bg     = C_FAIL_BG   if failed else C_PASS_BG
                fg     = C_FAIL_TEXT if failed else C_PASS_TEXT

                self.set_fill_color(*bg)
                self.set_text_color(*fg)
                self.set_font("Arial", "B" if failed else "", 8.5)

                badge  = "[FAIL]\" if failed else \"[PASS]"
                body   = (
                    f"  {badge}   Strategy: {stat.strategy_name}   |   "
                    f"Wait (mean / max): {stat.mean_wait_time_minutes:.1f} / "
                    f"{stat.max_wait_time_minutes:.1f} min   |   "
                    f"Avg session: {stat.avg_session_time_minutes:.0f} min   |   "
                    f"Truncated: {stat.sessions_truncated_count}   |   "
                    f"Failed patients: {stat.failed_patients_count}"
                )
                self.set_x(MARGIN_L)
                self.cell(BODY_W, 8, body, ln=1, fill=True)

            self.set_text_color(*C_TEXT)
            self.ln(3)

    # ─── Plot helper ──────────────────────────────────────────────────────────

    def _save_fig_to_temp(self, fig) -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        return path

    def add_plot(self, fig, w=BODY_W, center=True, caption_text: str = None):
        img_path = self._save_fig_to_temp(fig)
        x = (PAGE_W - w) / 2 if center else MARGIN_L
        self.image(img_path, x=x, w=w)
        os.remove(img_path)
        self.ln(2)
        if caption_text:
            self.caption(caption_text)


# ═══════════════════════════════════════════════════════════════════════════════
# Report generators
# ═══════════════════════════════════════════════════════════════════════════════

def generate_individual_report(
    strategy_name: str,
    results: List[ShiftStatistics],
    edge_cases: Dict[str, List[ShiftStatistics]],
    config: SimulationConfig,
    n_iterations: int,
    output_path: str,
):
    pdf = PDFReporter()
    viz = Visualizer()
    df  = pd.DataFrame([viz._stats_to_dict(r) for r in results])

    strat_results = [r for r in results if r.strategy_name == strategy_name]
    if not strat_results:
        return

    all_strategies = list(set(r.strategy_name for r in results))

    pdf.create_cover_page(
        title=f"Strategy Profile: {strategy_name}",
        subtitle="Isolated Performance and Tail-Risk Analytics",
        n_iterations=n_iterations,
        n_strategies=len(all_strategies),
        config=config,
    )

    pdf.add_page()
    pdf.write_config_section(config, n_iterations)

    pdf.chapter_title("Statistical Summary")
    pdf.chapter_body("Distributional metrics across all Monte Carlo iterations for this strategy.")
    pdf.draw_percentile_table(df, strategy_name)

    pdf.chapter_title("Distribution & Utilisation")
    fig_wait = viz.plot_single_wait_distribution(strategy_name, strat_results)
    pdf.add_plot(fig_wait, caption_text="Figure 1. Wait-time distribution across Monte Carlo runs.")

    fig_util = viz.plot_single_utilization(strategy_name, strat_results)
    pdf.add_plot(fig_util, w=130, caption_text="Figure 2. Machine and nurse utilisation.")

    pdf.add_page()
    pdf.chapter_title("Advanced Insights")
    fig_cdf = viz.plot_cdf_wait_time(results, strategy_name)
    pdf.add_plot(fig_cdf, w=155, caption_text="Figure 3. Empirical CDF of mean wait time.")

    fig_session = viz.plot_single_session_distribution(strategy_name, strat_results)
    pdf.add_plot(fig_session, w=155, caption_text="Figure 4. Actual session duration distribution (3 h min / 4 h nominal reference lines).")

    pdf.add_page()
    pdf.write_edge_cases(edge_cases, strategy_name)

    pdf.output(output_path)


def generate_global_comparison_report(
    results: List[ShiftStatistics],
    edge_cases: Dict[str, List[ShiftStatistics]],
    config: SimulationConfig,
    n_iterations: int,
    output_path: str,
):
    pdf = PDFReporter()
    viz = Visualizer()
    df  = pd.DataFrame([viz._stats_to_dict(r) for r in results])

    strategies = list(set(r.strategy_name for r in results))

    pdf.create_cover_page(
        title="Global Comparison Report",
        subtitle="Monte Carlo Strategy Paired-Difference Analytics",
        n_iterations=n_iterations,
        n_strategies=len(strategies),
        config=config,
    )

    pdf.add_page()
    pdf.write_config_section(config, n_iterations)

    pdf.chapter_title("Global Statistical Summary")
    pdf.chapter_body(
        "Per-strategy distributional metrics (mean +/- SD, p50, p90, p99) aggregated "
        "across the full Monte Carlo suite.  The p90 and p99 columns serve as "
        "tail-risk indicators; lower is better."
    )
    for strat in strategies:
        pdf.subsection_title(strat)
        pdf.draw_percentile_table(df, strat)

    pdf.add_page()
    pdf.chapter_title("Comparative Analytics")
    try:
        fig_paired = viz.plot_paired_difference(results, "mean_wait_time_minutes", "Mean Wait Time")
        pdf.add_plot(fig_paired, caption_text="Figure 1. Paired-difference analysis of mean wait time across strategies.")
    except Exception:
        pass

    fig_dist = viz.plot_wait_distribution(results)
    pdf.add_plot(fig_dist, caption_text="Figure 2. Wait-time distribution comparison across strategies.")

    pdf.add_page()
    fig_util_global = viz.plot_global_utilization_violin(results)
    pdf.add_plot(fig_util_global, caption_text="Figure 3. Utilisation violin plots across strategies.")

    fig_cdf = viz.plot_cdf_wait_time(results)
    pdf.add_plot(fig_cdf, caption_text="Figure 4. Empirical CDF of mean wait time -- all strategies.")

    pdf.add_page()
    fig_scatter = viz.plot_scatter_wait_vs_session_time(results)
    pdf.add_plot(fig_scatter, caption_text="Figure 5. Wait time vs. avg session duration -- all strategies.")

    pdf.write_edge_cases(edge_cases)

    pdf.output(output_path)
