import streamlit as st
import os
import json
import pandas as pd
from typing import List, Dict, Tuple

from src.config import SimulationConfig, IntRange, UniformIntSampler
from src.models import ShiftScenario, ShiftStatistics
from src.batcher.batcher import MonteCarloBatcher
from src.visualizer.visualizer import Visualizer
from presets_manager import list_presets, save_preset, load_preset, delete_preset

st.set_page_config(
    page_title="Dialysis Simulation Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ─── Defaults & helpers ───────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "strategies":    ["FIFO", "FIXED"],
    "n_iterations":  100,
    "global_seed":   42,
    "patient_vol":   (15, 20),
    "arrival_range": (0, 60),
    "setup_range":   (10, 20),
    "session_range": (210, 240),
    "min_session":   180,
    "machine_count": (15, 20),
    "machine_ready": (0, 90),
    "defect_chance": 0.15,
    "nurse_count":   (2, 4),
}

def _get(key):
    return st.session_state.get(f"cfg_{key}", _DEFAULTS[key])

def _collect_config_dict() -> dict:
    return {
        "strategies":    st.session_state.get("cfg_strategies",    _DEFAULTS["strategies"]),
        "n_iterations":  st.session_state.get("cfg_n_iterations",  _DEFAULTS["n_iterations"]),
        "global_seed":   st.session_state.get("cfg_global_seed",   _DEFAULTS["global_seed"]),
        "patient_vol":   list(st.session_state.get("cfg_patient_vol",    _DEFAULTS["patient_vol"])),
        "arrival_range": list(st.session_state.get("cfg_arrival_range",  _DEFAULTS["arrival_range"])),
        "setup_range":   list(st.session_state.get("cfg_setup_range",    _DEFAULTS["setup_range"])),
        "session_range": list(st.session_state.get("cfg_session_range",  _DEFAULTS["session_range"])),
        "min_session":   st.session_state.get("cfg_min_session",   _DEFAULTS["min_session"]),
        "machine_count": list(st.session_state.get("cfg_machine_count",  _DEFAULTS["machine_count"])),
        "machine_ready": list(st.session_state.get("cfg_machine_ready",  _DEFAULTS["machine_ready"])),
        "defect_chance": st.session_state.get("cfg_defect_chance", _DEFAULTS["defect_chance"]),
        "nurse_count":   list(st.session_state.get("cfg_nurse_count",    _DEFAULTS["nurse_count"])),
    }

def _apply_preset(data: dict) -> None:
    mapping = {
        "strategies":    ("cfg_strategies",    lambda v: v),
        "n_iterations":  ("cfg_n_iterations",  int),
        "global_seed":   ("cfg_global_seed",   int),
        "patient_vol":   ("cfg_patient_vol",   tuple),
        "arrival_range": ("cfg_arrival_range", tuple),
        "setup_range":   ("cfg_setup_range",   tuple),
        "session_range": ("cfg_session_range", tuple),
        "min_session":   ("cfg_min_session",   int),
        "machine_count": ("cfg_machine_count", tuple),
        "machine_ready": ("cfg_machine_ready", tuple),
        "defect_chance": ("cfg_defect_chance", float),
        "nurse_count":   ("cfg_nurse_count",   tuple),
    }
    for key, (sk, cast) in mapping.items():
        if key in data:
            st.session_state[sk] = cast(data[key])

# ─── Edge case detail card ────────────────────────────────────────────────────
def render_edge_case_card(
    case_name: str,
    scenario: ShiftScenario,
    stats_list: List[ShiftStatistics],
    strategy_filter: str = None,
):
    """Render one edge-case expandable card with full shift snapshot + results."""
    n_active    = len(scenario.machine_ready_times)
    n_defective = len(scenario.defective_machine_ids)
    n_patients  = len(scenario.patient_arrivals)
    arrivals    = [p["arrival_min"] for p in scenario.patient_arrivals]
    arr_lo, arr_hi = (min(arrivals), max(arrivals)) if arrivals else (0, 0)

    # Aggregate failure status for title badge
    filtered = [s for s in stats_list if not strategy_filter or s.strategy_name == strategy_filter]
    any_failed = any(s.failed_patients_count > 0 for s in filtered)
    badge = "🔴" if any_failed else "🟢"

    with st.expander(f"{badge} {case_name}", expanded=any_failed):
        snap_col, res_col = st.columns([1, 2])

        with snap_col:
            st.markdown("**📋 Shift Snapshot**")
            snap_data = {
                "Field": [
                    "Active machines", "Defective machines", "Total machines",
                    "Nurses", "Patients",
                    "Arrival window", "Shift duration",
                    "Min viable session", "Machine cooldown",
                    "Seed",
                ],
                "Value": [
                    n_active, n_defective, n_active + n_defective,
                    scenario.nurse_count, n_patients,
                    f"{arr_lo}–{arr_hi} min",
                    f"{scenario.shift_end_minutes} min",
                    f"{scenario.min_session_duration_minutes} min",
                    f"{scenario.machine_cooldown_minutes} min",
                    scenario.scenario_seed,
                ],
            }
            st.dataframe(
                pd.DataFrame(snap_data).set_index("Field"),
                use_container_width=True,
                height=360,
            )

            # Show defective machine IDs if any
            if scenario.defective_machine_ids:
                st.caption(f"Defective machine IDs: {sorted(scenario.defective_machine_ids)}")

            # Machine readiness delays
            delays = {
                mid: rt for mid, rt in scenario.machine_ready_times.items() if rt > 0
            }
            if delays:
                st.caption(f"Delayed machines: { {k: f'{v} min' for k, v in sorted(delays.items())} }")

        with res_col:
            st.markdown("**📊 Strategy Results**")
            for stat in filtered:
                failed = stat.failed_patients_count > 0
                color  = "#fff5f5" if failed else "#f0fff4"
                border = "#fc8181" if failed else "#68d391"
                icon   = "❌ FAIL" if failed else "✅ PASS"

                fail_rate = (stat.failed_patients_count / stat.total_patients_processed * 100) if stat.total_patients_processed else 0.0

                st.markdown(
                    f"""
                    <div style="
                        background:{color};border-left:4px solid {border};
                        border-radius:6px;padding:10px 14px;margin-bottom:10px;
                    ">
                        <b>{icon} &nbsp; {stat.strategy_name}</b><br>
                        <table style="width:100%;font-size:0.85em;margin-top:6px;border-collapse:collapse;">
                            <tr>
                                <td style="padding:2px 8px 2px 0"><b>Patients processed</b></td>
                                <td style="padding:2px 0">{stat.total_patients_processed}</td>
                                <td style="padding:2px 8px 2px 16px"><b>Failed</b></td>
                                <td style="padding:2px 0;color:{'#c53030' if failed else 'inherit'}">{stat.failed_patients_count} ({fail_rate:.1f}%)</td>
                            </tr>
                            <tr>
                                <td style="padding:2px 8px 2px 0"><b>Avg session</b></td>
                                <td style="padding:2px 0">{stat.avg_session_time_minutes:.1f} min</td>
                                <td style="padding:2px 8px 2px 16px"><b>Truncated</b></td>
                                <td style="padding:2px 0">{stat.sessions_truncated_count}</td>
                            </tr>
                            <tr>
                                <td style="padding:2px 8px 2px 0"><b>Mean wait</b></td>
                                <td style="padding:2px 0">{stat.mean_wait_time_minutes:.1f} min</td>
                                <td style="padding:2px 8px 2px 16px"><b>Max wait</b></td>
                                <td style="padding:2px 0">{stat.max_wait_time_minutes:.1f} min</td>
                            </tr>
                            <tr>
                                <td style="padding:2px 8px 2px 0"><b>Nurse util</b></td>
                                <td style="padding:2px 0">{stat.nurse_utilization_percent*100:.1f}%</td>
                                <td style="padding:2px 8px 2px 16px"><b>Machine util</b></td>
                                <td style="padding:2px 0">{stat.machine_utilization_percent*100:.1f}%</td>
                            </tr>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Dialysis Unit Simulation Dashboard")
st.markdown("Interactively execute Monte Carlo paired-difference tests across patient scheduling strategies.")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Preset Manager ────────────────────────────────────────────────────
    st.header("🗂️ Preset Configs")
    saved = list_presets()

    col_load, col_del = st.columns([3, 1])
    with col_load:
        selected_preset = st.selectbox(
            "Saved presets", options=["— select —"] + saved,
            key="preset_select", label_visibility="collapsed"
        )
    with col_del:
        del_clicked = st.button("🗑️", help="Delete selected preset", use_container_width=True)

    load_clicked = st.button(
        "⬇ Load preset", use_container_width=True,
        disabled=(selected_preset == "— select —")
    )

    if load_clicked and selected_preset != "— select —":
        data = load_preset(selected_preset)
        if data:
            _apply_preset(data)
            st.success(f'Loaded "{selected_preset}"')
            st.rerun()
        else:
            st.error("Preset file not found.")

    if del_clicked:
        if selected_preset == "— select —":
            st.warning("Select a preset to delete.")
        elif delete_preset(selected_preset):
            st.success(f'Deleted "{selected_preset}"')
            st.rerun()

    with st.expander("💾 Save current config as preset"):
        preset_name = st.text_input("Preset name", placeholder="e.g. High-volume stress test")
        if st.button("Save", use_container_width=True):
            if not preset_name.strip():
                st.error("Enter a preset name.")
            else:
                save_preset(preset_name.strip(), _collect_config_dict())
                st.success(f'Saved "{preset_name.strip()}"')
                st.rerun()

    st.divider()

    # ── Simulation Settings ───────────────────────────────────────────────
    st.header("Simulation Settings")
    strategies = st.multiselect(
        "Strategies to Compare", options=["FIFO", "FIXED"],
        default=_get("strategies"), key="cfg_strategies"
    )
    n_iterations = st.number_input(
        "Monte Carlo Iterations", min_value=1, max_value=5000,
        value=_get("n_iterations"), key="cfg_n_iterations"
    )
    global_seed = st.number_input(
        "Global Seed", min_value=0, value=_get("global_seed"), key="cfg_global_seed"
    )

    # ── Patient Constraints ───────────────────────────────────────────────
    st.header("Patient Constraints")
    patient_vol_min, patient_vol_max = st.slider(
        "Patient Volume Range", 5, 50, value=_get("patient_vol"), key="cfg_patient_vol"
    )
    arrival_min, arrival_max = st.slider(
        "Arrival Time Range (min)", 0, 180, value=_get("arrival_range"), key="cfg_arrival_range"
    )
    setup_min, setup_max = st.slider(
        "Nurse Setup Time Range (min)", 1, 60, value=_get("setup_range"), key="cfg_setup_range"
    )
    st.markdown("**Session Duration** — Prescribed per-patient dialysis time (~4 h target)")
    session_min, session_max = st.slider(
        "Prescribed Session Range (min)", 60, 360,
        value=_get("session_range"), key="cfg_session_range",
        help="Nominal session length per patient."
    )
    min_session = st.slider(
        "Minimum Viable Session (min)", 60, 300,
        value=_get("min_session"), key="cfg_min_session",
        help="Sessions shorter than this (due to late start) count as failed."
    )

    # ── Resource Constraints ──────────────────────────────────────────────
    st.header("Resource Constraints")
    machine_count_min, machine_count_max = st.slider(
        "Machine Count Range", 5, 50, value=_get("machine_count"), key="cfg_machine_count"
    )
    machine_ready_min, machine_ready_max = st.slider(
        "Machine Ready Delay (min)", 0, 120, value=_get("machine_ready"), key="cfg_machine_ready"
    )
    defect_chance = st.slider(
        "Defective Machine Chance", 0.0, 1.0,
        value=float(_get("defect_chance")), step=0.01, key="cfg_defect_chance"
    )

    # ── Personnel Constraints ─────────────────────────────────────────────
    st.header("Personnel Constraints")
    nurse_count_min, nurse_count_max = st.slider(
        "Nurse Count Range", 1, 10, value=_get("nurse_count"), key="cfg_nurse_count"
    )

    st.divider()
    run_clicked = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
if run_clicked:
    if len(strategies) == 0:
        st.error("Please select at least one strategy.")
        st.stop()

    with st.spinner("Running Monte Carlo simulation and edge cases..."):
        config = SimulationConfig(
            patient_volume=IntRange(patient_vol_min, patient_vol_max),
            total_machines=IntRange(machine_count_min, machine_count_max),
            nurse_count=IntRange(nurse_count_min, nurse_count_max),
            machine_ready_delay_minutes=IntRange(machine_ready_min, machine_ready_max),
            session_duration_minutes_range=IntRange(session_min, session_max),
            min_session_duration_minutes=min_session,
            machine_defect_probability=defect_chance
        )
        config = config.with_overrides(
            arrival_minute_sampler=UniformIntSampler(IntRange(arrival_min, arrival_max)),
            setup_duration_minutes_sampler=UniformIntSampler(IntRange(setup_min, setup_max))
        )
        try:
            config.validate()
        except ValueError as e:
            st.error(f"Configuration Error: {e}")
            st.stop()

        batcher = MonteCarloBatcher(
            config=config, strategy_ids=strategies,
            n_iterations=n_iterations, global_seed=global_seed
        )

        progress_bar = st.progress(0, text="Running Monte Carlo iterations...")
        def update_progress(current: int, total: int):
            progress_bar.progress(current / total, text=f"Running Monte Carlo iterations... ({current}/{total})")

        output_csv_path = "outputs/simulation_results.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        # Main run — streams to CSV and captures failed shifts
        _, auto_edge_cases = batcher.run_with_scenarios(
            output_csv_path=output_csv_path,
            progress_callback=update_progress
        )

        # Predefined edge cases
        predefined_edge_cases = batcher.edge_case_run()

        # Load streamed results into memory-efficient DataFrame
        results_df = pd.read_csv(output_csv_path)

        progress_bar.empty()

        # Merge: predefined first, then auto-discovered
        all_edge_cases = {**predefined_edge_cases, **auto_edge_cases}

        if "figures" in st.session_state:
            del st.session_state["figures"]

        st.session_state.update({
            "results":            results_df,
            "edge_cases":         all_edge_cases,
            "auto_edge_cases":    auto_edge_cases,
            "predefined_edge_cases": predefined_edge_cases,
            "config":             config,
            "n_iterations":       n_iterations,
            "strategies":         strategies,
        })

    n_auto = len(auto_edge_cases)
    st.success(
        f"Simulation completed: {len(results_df)} runs "
        f"({n_iterations} iter x {len(strategies)} strategies) + "
        f"{len(predefined_edge_cases)} predefined edge cases + "
        f"{n_auto} auto-captured boundary shift{'s' if n_auto != 1 else ''}"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if "results" in st.session_state:
    results              = st.session_state["results"]
    all_edge_cases       = st.session_state["edge_cases"]
    auto_edge_cases      = st.session_state["auto_edge_cases"]
    predefined_edge_cases= st.session_state["predefined_edge_cases"]
    config               = st.session_state["config"]
    n_iterations         = st.session_state["n_iterations"]
    strategies           = st.session_state["strategies"]

    # ── KPI Cards ─────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators (Aggregated Averages)")
    
    viz = Visualizer()
    df = viz._ensure_dataframe(results).copy()
    df["failure_rate_percent"] = (df["failed_patients_count"] / df["total_patients_processed"]) * 100
    means = df.groupby("strategy_name").mean(numeric_only=True)

    cols = st.columns(len(strategies))
    for idx, strat in enumerate(strategies):
        with cols[idx]:
            st.markdown(f"**{strat}**")
            d = means.loc[strat] if strat in means.index else None
            if d is not None:
                st.metric("Avg Mean Wait (min)", f"{d['mean_wait_time_minutes']:.1f}")
                st.metric("Avg Max Wait (min)",  f"{d['max_wait_time_minutes']:.1f}")
                st.metric("Avg Session Time (min)", f"{d['avg_session_time_minutes']:.1f}",
                          help="Mean actual dialysis session duration")
                st.metric("Avg Truncated Sessions", f"{d['sessions_truncated_count']:.1f}",
                          help="Sessions cut short by shift wall")
                st.metric("Avg Failed Patients", f"{d['failed_patients_count']:.1f}")
                st.metric("Failure Rate", f"{d['failure_rate_percent']:.2f}%")
                st.metric("Nurse Utilization",   f"{d['nurse_utilization_percent']*100:.1f}%")
                st.metric("Machine Utilization", f"{d['machine_utilization_percent']*100:.1f}%")

    st.divider()

    st.subheader("Simulation Analytics")

    if "figures" not in st.session_state:
        st.session_state["figures"] = {}
        prog = st.progress(0, text="Generating Wait Time Distribution...")
        
        st.session_state["figures"]["wait_dist"] = viz.plot_wait_distribution(results)
        prog.progress(1/7, text="Generating Session Time Distribution...")
        
        st.session_state["figures"]["session_dist"] = viz.plot_session_time_distribution(results)
        prog.progress(2/7, text="Generating Resource Utilization...")
        
        st.session_state["figures"]["util"] = viz.plot_utilization(results)
        prog.progress(3/7, text="Generating Paired Difference...")
        
        if len(strategies) >= 2:
            try:
                st.session_state["figures"]["paired"] = viz.plot_paired_difference(results)
            except ValueError:
                pass
        
        prog.progress(4/7, text="Generating Wait Time Series...")
        st.session_state["figures"]["iter_wait"] = viz.plot_metric_over_iterations(results, "mean_wait_time_minutes", "Mean Wait Time vs Iteration", "Wait Time (min)")
        
        prog.progress(5/7, text="Generating Max Wait Series...")
        st.session_state["figures"]["iter_max"] = viz.plot_metric_over_iterations(results, "max_wait_time_minutes",  "Max Wait Time vs Iteration",  "Wait Time (min)")
        
        prog.progress(6/7, text="Generating Session Time Series...")
        st.session_state["figures"]["iter_session"] = viz.plot_metric_over_iterations(results, "avg_session_time_minutes", "Avg Session Time vs Iteration", "Duration (min)")
        
        prog.progress(1.0, text="Visualizations ready!")
        prog.empty()

    figs = st.session_state["figures"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Wait Time Distribution")
        st.pyplot(figs["wait_dist"])

        st.markdown("#### Session Time Distribution")
        st.pyplot(figs["session_dist"])

    with col2:
        st.markdown("#### Resource Utilization")
        st.pyplot(figs["util"])

        if len(strategies) >= 2 and "paired" in figs:
            st.markdown("#### Paired Difference (Mean Wait)")
            st.pyplot(figs["paired"])

    st.divider()
    st.subheader("Time Series Analysis (Across Iterations)")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.pyplot(figs["iter_wait"])
    with col4:
        st.pyplot(figs["iter_max"])
    with col5:
        st.pyplot(figs["iter_session"])

    st.divider()

    # ── Edge Cases ────────────────────────────────────────────────────────
    st.subheader("Edge-Case Stress Tests")

    n_auto = len(auto_edge_cases)
    if n_auto:
        st.info(
            f"**{n_auto} boundary shifts** automatically captured. "
            f"The simulation tracks the absolute Best Case and Worst Case scenario for each strategy "
            f"to highlight the extremes without overloading memory.",
            icon="💡"
        )

    tab_labels = []
    if predefined_edge_cases:
        tab_labels.append(f"📋 Predefined ({len(predefined_edge_cases)})")
    if auto_edge_cases:
        tab_labels.append(f"🔍 Auto-captured ({n_auto})")
    tab_labels.append("📁 All")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    if predefined_edge_cases:
        with tabs[tab_idx]:
            for name, (scenario, stats) in predefined_edge_cases.items():
                render_edge_case_card(name, scenario, stats)
        tab_idx += 1

    if auto_edge_cases:
        with tabs[tab_idx]:
            st.caption(
                "These cards represent the absolute Best Case (fewest failures/lowest wait) and Worst Case "
                "(most failures/highest wait) boundary iterations for each strategy."
            )
            for name, (scenario, stats) in auto_edge_cases.items():
                render_edge_case_card(name, scenario, stats)
        tab_idx += 1

    # All tab
    with tabs[tab_idx]:
        for name, (scenario, stats) in all_edge_cases.items():
            render_edge_case_card(name, scenario, stats)

    st.divider()

    # ── PDF Reports ───────────────────────────────────────────────────────
    st.subheader("PDF Reports")
    st.markdown("Generate comprehensive PDF reports covering configuration, strategy charts, comparative graphics, and edge-case behaviors.")

    from src.reporter.pdf_reporter import generate_individual_report, generate_global_comparison_report
    import tempfile

    report_cols = st.columns(2)
    with report_cols[0]:
        st.markdown("##### Individual Strategy Reports")
        strat_to_report = st.selectbox("Select Strategy", options=strategies)
        if st.button(f"Generate {strat_to_report} Report"):
            with st.spinner(f"Generating PDF for {strat_to_report}..."):
                fd, path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                generate_individual_report(
                    strat_to_report, results, all_edge_cases, config, n_iterations, path
                )
                with open(path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label=f"Download {strat_to_report} Report",
                    data=pdf_bytes,
                    file_name=f"dialysis_report_{strat_to_report}.pdf",
                    mime="application/pdf"
                )

    with report_cols[1]:
        st.markdown("##### Global Comparison Report")
        if st.button("Generate Global Report", type="primary"):
            with st.spinner("Generating Global PDF Report..."):
                fd, path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                generate_global_comparison_report(
                    results, all_edge_cases, config, n_iterations, path
                )
                with open(path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="Download Global Comparison Report",
                    data=pdf_bytes,
                    file_name="dialysis_global_comparison_report.pdf",
                    mime="application/pdf"
                )

else:
    st.info("Adjust settings in the sidebar and click 'Run Simulation' to begin.")
