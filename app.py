from __future__ import annotations

import os
import sys

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from simulation import generate_traffic_matrix, generate_vms, load_uploaded_data, run_simulation, save_results_csv
from visualization import generate_all_charts


def main() -> None:
    """Run the optional Streamlit dashboard when Streamlit is installed."""

    if st is None:
        print("Streamlit is not installed in this Python environment.")
        print("Use the production fallback dashboard instead:")
        print("  python local_dashboard.py")
        print("Then open http://localhost:8501")
        return

    st.set_page_config(page_title="TA-MOACO VM Placement Demo", layout="wide")

    st.title("TA-MOACO for Data Center VM Placement")
    st.caption("Faculty presentation demo: server power, network power, traffic locality, and VM placement.")

    with st.sidebar:
        st.header("Simulation")
        mode = st.radio("Data source", ["Generate synthetic data", "Upload CSV files"])
        num_pms = st.number_input("Physical machines", min_value=20, max_value=300, value=100, step=10)
        num_vms = st.slider("Virtual machines", min_value=200, max_value=800, value=200, step=50)
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
        ants = st.slider("ACO ants", min_value=10, max_value=80, value=50, step=10)
        iterations = st.slider("TA-MOACO iterations", min_value=5, max_value=40, value=15, step=5)
        alpha = st.slider("TA-MOACO alpha", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        beta = st.slider("TA-MOACO beta", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
        evaporation = st.slider("TA-MOACO evaporation", min_value=0.01, max_value=0.5, value=0.1, step=0.01)

        vm_file = None
        traffic_file = None
        if mode == "Upload CSV files":
            vm_file = st.file_uploader("VM CSV: id,cpu,ram", type=["csv"])
            traffic_file = st.file_uploader("Traffic CSV: src_vm,dst_vm,demand", type=["csv"])

        run_clicked = st.button("Run algorithms", type="primary")

    with st.expander("How TA-MOACO works", expanded=True):
        st.markdown(
            """
            1. Each ant builds a complete VM-to-PM placement while respecting CPU and RAM capacity.
            2. Candidate PM probability uses pheromone, a resource heuristic, and a traffic-affinity heuristic.
            3. Resource heuristic favors balanced, well-packed servers so idle PMs can be switched off.
            4. Traffic-affinity heuristic favors placing communicating VMs on the same PM, same ToR, or same pod.
            5. After each iteration, pheromone evaporates and elite placements deposit new pheromone.
            6. The selected placement minimizes server power, network power, active PMs, and average hop count.
            """
        )

    if not run_clicked:
        st.info("Choose settings in the sidebar and run the algorithms.")
        return

    try:
        if mode == "Upload CSV files":
            if vm_file is None:
                st.error("Please upload a VM CSV file.")
                st.stop()
            vms, traffic = load_uploaded_data(vm_file, traffic_file)
        else:
            vms = generate_vms(num_vms, seed=seed)
            traffic = generate_traffic_matrix(num_vms, seed=seed)

        with st.spinner("Running FFD, ACS-VMP, Traffic-Aware VMP, and TA-MOACO..."):
            df, results, data_center = run_simulation(
                num_pms=int(num_pms),
                num_vms=len(vms),
                seed=int(seed),
                ants=int(ants),
                iterations=int(iterations),
                alpha=float(alpha),
                beta=float(beta),
                evaporation=float(evaporation),
                vms=vms,
                traffic=traffic,
            )
            csv_path = save_results_csv(df)
            chart_paths = generate_all_charts(df, results, data_center, traffic=traffic)

        st.success("Simulation complete.")

        best = min(results, key=lambda result: result.total_power)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Recommended", best.name)
        metric_cols[1].metric("Total power", f"{best.total_power:.0f} W")
        metric_cols[2].metric("Energy savings", f"{best.energy_savings_pct:.2f}%")
        metric_cols[3].metric("Average hops", f"{best.average_hop_count:.2f}")

        st.subheader("Comparison Table")

        def highlight_best(row):
            return ["background-color: #e8fff5; font-weight: 700" if row.get("best_algorithm") == "YES" else "" for _ in row]

        st.dataframe(df.style.apply(highlight_best, axis=1), use_container_width=True)
        st.download_button("Download results CSV", df.to_csv(index=False), file_name="results.csv", mime="text/csv")

        st.subheader("Presentation Graphs")
        graph_cols = st.columns(2)
        for index, path in enumerate(chart_paths):
            with graph_cols[index % 2]:
                st.image(path, use_container_width=True)

        st.subheader("Final Recommended VM Placement")
        placement_df = pd.DataFrame(
            [{"vm_id": vm_id, "pm_id": pm_id} for vm_id, pm_id in sorted(best.placement.items())]
        )
        st.dataframe(placement_df.head(300), use_container_width=True)
        st.caption(f"Showing first 300 assignments. Full placement contains {len(placement_df)} VMs.")
        st.caption(f"Results saved to {os.path.abspath(csv_path)}")
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
