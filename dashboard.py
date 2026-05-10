import streamlit as st
import os
import json
import pandas as pd
from typing import List

from src.config import SimulationConfig, IntRange, UniformIntSampler
from src.batcher.batcher import MonteCarloBatcher
from src.visualizer.visualizer import Visualizer
from presets_manager import list_presets, save_preset, load_preset, delete_preset

st.set_page_config(
    page_title="Dialysis Simulation Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ─── Preset key → (default_value) mapping ────────────────────────────────────
_DEFAULTS: dict = {
    "strategies":       ["FIFO", "FIXED"],
    "n_iterations":     100,
    "global_seed":      42,
    "patient_vol":      (15, 20),
    "arrival_range":    (0, 60),
    "setup_range":      (10, 20),
    "session_range":    (210, 240),
    "min_session":      180,
    "machine_count":    (15, 20),
    "machine_ready":    (0, 90),
    "defect_chance":    0.15,
    "nurse_count":      (2, 4),
}

def _get(key):
    """Return session_state value if set (by preset loader), else the default."""
    return st.session_state.get(f"cfg_{key}", _DEFAULTS[key])

def _collect_config_dict() -> dict:
    """Snapshot all current widget values into a JSON-serialisable dict."""
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
    """Write preset values into session_state so widgets pick them up on rerun."""
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

# ─── Page header ─────────────────────────────────────────────────────────────
st.title("Dialysis Unit Simulation Dashboard")
st.markdown("Interactively execute Monte Carlo paired-difference tests across patient scheduling strategies.")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Preset Manager ────────────────────────────────────────────────────
    st.header("🗂️ Preset Configs")

    saved = list_presets()

    col_load, col_del = st.columns([3, 1])

    with col_load:
        selected_preset = st.selectbox(
            "Saved presets",
            options=["— select —"] + saved,
            key="preset_select",
            label_visibility="collapsed"
        )
    with col_del:
        del_clicked = st.button("🗑️", help="Delete selected preset", use_container_width=True)

    load_clicked = st.button("⬇ Load preset", use_container_width=True, disabled=(selected_preset == "— select —"))

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
        "Strategies to Compare",
        options=["FIFO", "FIXED"],
        default=_get("strategies"),
        key="cfg_strategies"
    )

    n_iterations = st.number_input(
        "Monte Carlo Iterations", min_value=1, max_value=1000,
        value=_get("n_iterations"), key="cfg_n_iterations"
    )
    global_seed = st.number_input(
        "Global Seed", min_value=0,
        value=_get("global_seed"), key="cfg_global_seed"
    )

    # ── Patient Constraints ───────────────────────────────────────────────
    st.header("Patient Constraints")

    patient_vol_min, patient_vol_max = st.slider(
        "Patient Volume Range", 5, 50,
        value=_get("patient_vol"), key="cfg_patient_vol"
    )
    arrival_min, arrival_max = st.slider(
        "Arrival Time Range (T=0 to Max)", 0, 180,
        value=_get("arrival_range"), key="cfg_arrival_range"
    )
    setup_min, setup_max = st.slider(
        "Nurse Setup Time Range (min)", 1, 60,
        value=_get("setup_range"), key="cfg_setup_range"
    )
    st.markdown("**Session Duration** — Prescribed per-patient dialysis time (~4 h target, min 3 h viable)")
    session_min, session_max = st.slider(
        "Prescribed Session Range (min)", 60, 360,
        value=_get("session_range"), key="cfg_session_range",
        help="Nominal session length per patient. Sessions cut by the 6-hour shift wall below the minimum are marked failed."
    )
    min_session = st.slider(
        "Minimum Viable Session (min)", 60, 300,
        value=_get("min_session"), key="cfg_min_session",
        help="Sessions shorter than this threshold (due to late start) count as failed."
    )

    # ── Resource Constraints ──────────────────────────────────────────────
    st.header("Resource Constraints")

    machine_count_min, machine_count_max = st.slider(
        "Machine Count Range", 5, 50,
        value=_get("machine_count"), key="cfg_machine_count"
    )
    machine_ready_min, machine_ready_max = st.slider(
        "Machine Ready Delay (min)", 0, 120,
        value=_get("machine_ready"), key="cfg_machine_ready"
    )
    defect_chance = st.slider(
        "Defective Machine Chance", 0.0, 1.0,
        value=float(_get("defect_chance")), step=0.01, key="cfg_defect_chance"
    )

    # ── Personnel Constraints ─────────────────────────────────────────────
    st.header("Personnel Constraints")

    nurse_count_min, nurse_count_max = st.slider(
        "Nurse Count Range", 1, 10,
        value=_get("nurse_count"), key="cfg_nurse_count"
    )

    st.divider()
    run_clicked = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════
if run_clicked:
    if len(strategies) == 0:
        st.error("Please select at least one strategy.")
        st.stop()

    with st.spinner("Running Monte Carlo simulation and Edge Cases..."):
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
            config=config,
            strategy_ids=strategies,
            n_iterations=n_iterations,
            global_seed=global_seed
        )

        results = batcher.run()
        edge_cases = batcher.edge_case_run()

        st.session_state['results']     = results
        st.session_state['edge_cases']  = edge_cases
        st.session_state['config']      = config
        st.session_state['n_iterations']= n_iterations
        st.session_state['strategies']  = strategies

    st.success(
        f"Simulation completed: {len(results)} runs "
        f"({n_iterations} iterations x {len(strategies)} strategies) + Edge Cases"
    )

# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════
if 'results' in st.session_state:
    results      = st.session_state['results']
    edge_cases   = st.session_state['edge_cases']
    config       = st.session_state['config']
    n_iterations = st.session_state['n_iterations']
    strategies   = st.session_state['strategies']

    # ── KPI Cards ─────────────────────────────────────────────────────────
    st.subheader("Key Performance Indicators (Aggregated Averages)")
    df = pd.DataFrame([Visualizer._stats_to_dict(r) for r in results])
    means = df.groupby("strategy_name").mean(numeric_only=True)

    cols = st.columns(len(strategies))
    for idx, strat in enumerate(strategies):
        with cols[idx]:
            st.markdown(f"**{strat}**")
            strat_data = means.loc[strat] if strat in means.index else None
            if strat_data is not None:
                st.metric("Avg Mean Wait (min)", f"{strat_data['mean_wait_time_minutes']:.1f}")
                st.metric("Avg Max Wait (min)",  f"{strat_data['max_wait_time_minutes']:.1f}")
                st.metric(
                    "Avg Session Time (min)",
                    f"{strat_data['avg_session_time_minutes']:.1f}",
                    help="Mean actual dialysis session duration across all served patients"
                )
                st.metric(
                    "Avg Truncated Sessions",
                    f"{strat_data['sessions_truncated_count']:.1f}",
                    help="Mean number of sessions cut short by the shift wall per iteration"
                )
                st.metric("Avg Failed Patients", f"{strat_data['failed_patients_count']:.1f}")
                n_util = strat_data['nurse_utilization_percent'] * 100
                m_util = strat_data['machine_utilization_percent'] * 100
                st.metric("Nurse Utilization",   f"{n_util:.1f}%")
                st.metric("Machine Utilization", f"{m_util:.1f}%")

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────
    st.subheader("Simulation Analytics")
    viz = Visualizer()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Wait Time Distribution")
        st.pyplot(viz.plot_wait_distribution(results))

        st.markdown("#### Session Time Distribution")
        st.pyplot(viz.plot_session_time_distribution(results))

    with col2:
        st.markdown("#### Resource Utilization")
        st.pyplot(viz.plot_utilization(results))

        if len(strategies) >= 2:
            st.markdown("#### Paired Difference (Mean Wait)")
            try:
                st.pyplot(viz.plot_paired_difference(results))
            except ValueError as e:
                st.warning(str(e))

    st.divider()
    st.subheader("Time Series Analysis (Across Iterations)")

    col3, col4, col5 = st.columns(3)
    with col3:
        st.pyplot(viz.plot_metric_over_iterations(
            results, "mean_wait_time_minutes",
            "Mean Wait Time vs Iteration", "Wait Time (min)"
        ))
    with col4:
        st.pyplot(viz.plot_metric_over_iterations(
            results, "max_wait_time_minutes",
            "Max Wait Time vs Iteration", "Wait Time (min)"
        ))
    with col5:
        st.pyplot(viz.plot_metric_over_iterations(
            results, "avg_session_time_minutes",
            "Avg Session Time vs Iteration", "Session Duration (min)"
        ))

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
                generate_individual_report(strat_to_report, results, edge_cases, config, n_iterations, path)
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
        st.markdown("Generates a combined document with all strategies and comparative metrics.")
        if st.button("Generate Global Report", type="primary"):
            with st.spinner("Generating Global PDF Report..."):
                fd, path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                generate_global_comparison_report(results, edge_cases, config, n_iterations, path)
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
