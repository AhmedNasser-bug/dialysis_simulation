import streamlit as st
import os
import random
import pandas as pd
from typing import List

from src.config import SimulationConfig, IntRange, UniformIntSampler
from src.batcher.batcher import MonteCarloBatcher
from src.visualizer.visualizer import Visualizer

st.set_page_config(
    page_title="Dialysis Simulation Dashboard",
    page_icon="🏥",
    layout="wide"
)

st.title("Dialysis Unit Simulation Dashboard")
st.markdown("Interactively execute Monte Carlo paired-difference tests across patient scheduling strategies.")

# Sidebar Configuration
st.sidebar.header("Simulation Settings")

strategies = st.sidebar.multiselect(
    "Strategies to Compare",
    options=["FIFO", "FIXED"],
    default=["FIFO", "FIXED"]
)

n_iterations = st.sidebar.number_input("Monte Carlo Iterations", min_value=1, max_value=1000, value=100)
global_seed = st.sidebar.number_input("Global Seed", min_value=0, value=42)

st.sidebar.header("Patient Constraints")
patient_vol_min, patient_vol_max = st.sidebar.slider("Patient Volume Range", 5, 50, (15, 20))
arrival_min, arrival_max = st.sidebar.slider("Arrival Time Range (T=0 to Max)", 0, 180, (0, 60))
setup_min, setup_max = st.sidebar.slider("Nurse Setup Time Range (min)", 1, 60, (10, 20))

st.sidebar.markdown(
    "**Session Duration** — Prescribed per-patient dialysis time (~4 h target, min 3 h viable)"
)
session_min, session_max = st.sidebar.slider(
    "Prescribed Session Range (min)", 60, 360, (210, 240),
    help="Nominal session length per patient. Sessions cut by the 6-hour shift wall below 3 hours are marked failed."
)
min_session = st.sidebar.slider(
    "Minimum Viable Session (min)", 60, 300, 180,
    help="Sessions shorter than this threshold (due to late start) count as failed."
)

st.sidebar.header("Resource Constraints")
machine_count_min, machine_count_max = st.sidebar.slider("Machine Count Range", 5, 50, (15, 20))
machine_ready_min, machine_ready_max = st.sidebar.slider("Machine Ready Delay (min)", 0, 120, (0, 90))
defect_chance = st.sidebar.slider("Defective Machine Chance", 0.0, 1.0, 0.15, step=0.01)

st.sidebar.header("Personnel Constraints")
nurse_count_min, nurse_count_max = st.sidebar.slider("Nurse Count Range", 1, 10, (2, 4))

if st.sidebar.button("▶ Run Simulation", type="primary"):
    if len(strategies) == 0:
        st.error("Please select at least one strategy.")
        st.stop()

    with st.spinner("Running Monte Carlo simulation and Edge Cases..."):
        # Build config
        config = SimulationConfig(
            patient_volume=IntRange(patient_vol_min, patient_vol_max),
            total_machines=IntRange(machine_count_min, machine_count_max),
            nurse_count=IntRange(nurse_count_min, nurse_count_max),
            machine_ready_delay_minutes=IntRange(machine_ready_min, machine_ready_max),
            session_duration_minutes_range=IntRange(session_min, session_max),
            min_session_duration_minutes=min_session,
            machine_defect_probability=defect_chance
        )

        # Override samplers
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

        st.session_state['results'] = results
        st.session_state['edge_cases'] = edge_cases
        st.session_state['config'] = config
        st.session_state['n_iterations'] = n_iterations
        st.session_state['strategies'] = strategies

    st.success(f"Simulation completed: {len(results)} runs ({n_iterations} iterations x {len(strategies)} strategies) + Edge Cases")

if 'results' in st.session_state:
    results = st.session_state['results']
    edge_cases = st.session_state['edge_cases']
    config = st.session_state['config']
    n_iterations = st.session_state['n_iterations']
    strategies = st.session_state['strategies']

    # Overview Metrics
    st.subheader("Key Performance Indicators (Aggregated Averages)")
    df = pd.DataFrame([Visualizer._stats_to_dict(r) for r in results])

    # Calculate means per strategy
    means = df.groupby("strategy_name").mean(numeric_only=True)

    cols = st.columns(len(strategies))
    for idx, strat in enumerate(strategies):
        with cols[idx]:
            st.markdown(f"**{strat}**")
            strat_data = means.loc[strat] if strat in means.index else None
            if strat_data is not None:
                st.metric("Avg Mean Wait (min)", f"{strat_data['mean_wait_time_minutes']:.1f}")
                st.metric("Avg Max Wait (min)", f"{strat_data['max_wait_time_minutes']:.1f}")
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

                # Format utilizations
                n_util = strat_data['nurse_utilization_percent'] * 100
                m_util = strat_data['machine_utilization_percent'] * 100
                st.metric("Nurse Utilization", f"{n_util:.1f}%")
                st.metric("Machine Utilization", f"{m_util:.1f}%")

    st.divider()

    # Visualizations
    st.subheader("Simulation Analytics")
    viz = Visualizer()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Wait Time Distribution")
        fig_wait = viz.plot_wait_distribution(results)
        st.pyplot(fig_wait)

        st.markdown("#### Session Time Distribution")
        fig_session = viz.plot_session_time_distribution(results)
        st.pyplot(fig_session)

    with col2:
        st.markdown("#### Resource Utilization")
        fig_util = viz.plot_utilization(results)
        st.pyplot(fig_util)

        if len(strategies) >= 2:
            st.markdown("#### Paired Difference (Mean Wait)")
            try:
                fig_paired = viz.plot_paired_difference(results)
                st.pyplot(fig_paired)
            except ValueError as e:
                st.warning(str(e))

    st.divider()
    st.subheader("Time Series Analysis (Across Iterations)")

    col3, col4, col5 = st.columns(3)

    with col3:
        fig_mean_iter = viz.plot_metric_over_iterations(
            results,
            "mean_wait_time_minutes",
            "Mean Wait Time vs Iteration",
            "Wait Time (min)"
        )
        st.pyplot(fig_mean_iter)

    with col4:
        fig_max_iter = viz.plot_metric_over_iterations(
            results,
            "max_wait_time_minutes",
            "Max Wait Time vs Iteration",
            "Wait Time (min)"
        )
        st.pyplot(fig_max_iter)

    with col5:
        fig_session_iter = viz.plot_metric_over_iterations(
            results,
            "avg_session_time_minutes",
            "Avg Session Time vs Iteration",
            "Session Duration (min)"
        )
        st.pyplot(fig_session_iter)

    st.divider()
    st.subheader("PDF Reports")
    st.markdown("Generate comprehensive PDF reports covering the configuration, isolated strategy charts, comparative graphics, and edge-case behaviors.")

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
