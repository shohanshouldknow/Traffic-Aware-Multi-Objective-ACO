from __future__ import annotations

import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import AlgorithmResult
from topology import active_switches_for_placement


FIG_DPI = 300
TEXT = "#162033"
MUTED = "#627089"
BLUE = "#246BFE"
GREEN = "#16845F"
TEAL = "#0EA5A3"
ORANGE = "#C67A12"
RED = "#D84A4A"
GRID = "#E6EDF5"


def _prepare(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D8E1EC",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 11,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _save(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _finish_axis(ax, ylabel: str | None = None) -> None:
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _bar_labels(ax, bars, suffix: str = "") -> None:
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.2f}" if abs(height) < 10 and suffix == "" else f"{height:.0f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{label}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT,
        )


def _best_colors(df: pd.DataFrame, default: str = BLUE, best: str = GREEN) -> List[str]:
    if "best_algorithm" not in df.columns:
        return [default] * len(df)
    return [best if value == "YES" else default for value in df["best_algorithm"]]


def _ta_moaco_pm_utilization(results: List[AlgorithmResult]) -> pd.DataFrame:
    result = next((item for item in results if item.name == "TA-MOACO"), results[-1])
    return pd.DataFrame(result.metadata.get("pm_utilization", []))


def plot_energy_comparison(df: pd.DataFrame, output_dir: str = "results") -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "energy_comparison.png")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    algorithms = df["algorithm"].tolist()
    x = range(len(algorithms))
    server = df["server_power_w"].astype(float).tolist()
    network = df["network_power_w"].astype(float).tolist()
    ax.bar(x, server, color=BLUE, label="Server power")
    ax.bar(x, network, bottom=server, color=ORANGE, label="Network power")
    for index, total in enumerate(df["total_power_w"].astype(float)):
        ax.text(index, total, f"{total:.0f} W", ha="center", va="bottom", fontsize=10, color=TEXT, fontweight="bold")
    ax.set_title("Total Data Center Power by Algorithm")
    ax.set_xlabel("")
    ax.set_xticks(list(x), algorithms, rotation=12)
    ax.legend()
    _finish_axis(ax, "Power (W)")
    return _save(fig, path)


def plot_server_network_power(df: pd.DataFrame, output_dir: str = "results") -> str:
    """Show server and network power as side-by-side bars for clear comparison."""

    _prepare(output_dir)
    path = os.path.join(output_dir, "server_network_power.png")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    algorithms = df["algorithm"].tolist()
    x = np.arange(len(algorithms))
    width = 0.36
    server_bars = ax.bar(x - width / 2, df["server_power_w"], width=width, color=BLUE, label="Server power")
    network_bars = ax.bar(x + width / 2, df["network_power_w"], width=width, color=ORANGE, label="Network power")
    _bar_labels(ax, server_bars)
    _bar_labels(ax, network_bars)
    ax.set_title("Server vs Network Power")
    ax.set_xticks(x, algorithms, rotation=12)
    ax.legend()
    _finish_axis(ax, "Power (W)")
    return _save(fig, path)


def plot_active_components(df: pd.DataFrame, output_dir: str = "results") -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "active_components.png")
    fig, ax = plt.subplots(figsize=(13, 7.5))
    algorithms = df["algorithm"].tolist()
    x = list(range(len(algorithms)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], df["active_pms"], width=width, color=GREEN, label="Active PMs")
    ax.bar([i + width / 2 for i in x], df["active_switches"], width=width, color=RED, label="Active switches")
    ax.set_title("Active Physical Machines and Switches")
    ax.set_xlabel("")
    ax.set_xticks(x, algorithms, rotation=12)
    ax.legend()
    _finish_axis(ax, "Active count")
    return _save(fig, path)


def plot_active_pm_comparison(df: pd.DataFrame, output_dir: str = "results") -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "active_pm_comparison.png")
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df["algorithm"], df["active_pms"], color=_best_colors(df, BLUE, GREEN), width=0.58)
    _bar_labels(ax, bars)
    ax.set_title("Active Physical Machine Comparison")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)
    _finish_axis(ax, "Active PMs")
    return _save(fig, path)


def plot_active_switch_comparison(df: pd.DataFrame, output_dir: str = "results") -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "active_switch_comparison.png")
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df["algorithm"], df["active_switches"], color=_best_colors(df, ORANGE, GREEN), width=0.58)
    _bar_labels(ax, bars)
    ax.set_title("Active Switch Comparison")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)
    _finish_axis(ax, "Active switches")
    return _save(fig, path)


def plot_hop_count(df: pd.DataFrame, output_dir: str = "results") -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "hop_count_comparison.png")
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df["algorithm"], df["average_hop_count"], color=_best_colors(df, TEAL, GREEN), width=0.58)
    _bar_labels(ax, bars)
    ax.set_title("Traffic-Weighted Average Hop Count")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)
    _finish_axis(ax, "Average hops")
    return _save(fig, path)


def plot_placement_map(results: List[AlgorithmResult], output_dir: str = "results") -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "placement_map.png")
    best = next((result for result in results if result.name == "TA-MOACO"), results[-1])
    pm_counts = pd.Series(best.placement).value_counts().sort_index()
    placement_df = pd.DataFrame({"pm_id": pm_counts.index, "vm_count": pm_counts.values})
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.bar(placement_df["pm_id"], placement_df["vm_count"], color=BLUE, width=0.86)
    ax.set_title("TA-MOACO VM Placement Map")
    ax.set_xlabel("Physical machine ID")
    step = max(1, len(placement_df) // 25)
    ticks = placement_df["pm_id"].tolist()[::step]
    ax.set_xticks(ticks)
    _finish_axis(ax, "VMs placed")
    return _save(fig, path)


def plot_topology_placement(results: List[AlgorithmResult], data_center, output_dir: str = "results") -> str:
    """Show the recommended VM placement as a PM grid grouped by ToR switch."""

    _prepare(output_dir)
    path = os.path.join(output_dir, "topology_placement.png")
    best = next((result for result in results if result.name == "TA-MOACO"), results[-1])
    vm_counts = pd.Series(best.placement).value_counts().to_dict()

    pms = sorted(data_center.pms, key=lambda pm: pm.id)
    pms_per_tor = max(1, max(sum(1 for pm in pms if pm.tor_id == tor_id) for tor_id in set(pm.tor_id for pm in pms)))
    num_tors = max(pm.tor_id for pm in pms) + 1

    grid = []
    labels = []
    for tor_id in range(num_tors):
        row = []
        label_row = []
        tor_pms = [pm for pm in pms if pm.tor_id == tor_id]
        for slot in range(pms_per_tor):
            if slot < len(tor_pms):
                pm = tor_pms[slot]
                row.append(vm_counts.get(pm.id, 0))
                label_row.append(f"PM{pm.id}\n{vm_counts.get(pm.id, 0)} VM")
            else:
                row.append(float("nan"))
                label_row.append("")
        grid.append(row)
        labels.append(label_row)

    fig, ax = plt.subplots(figsize=(14, max(6, num_tors * 0.65)))
    image = ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    ax.set_title("TA-MOACO VM Placement by ToR Switch")
    ax.set_xlabel("PM slot under each ToR")
    ax.set_ylabel("ToR / edge switch")
    ax.set_yticks(range(num_tors), [f"ToR {i}" for i in range(num_tors)])
    ax.set_xticks(range(pms_per_tor), [f"Slot {i}" for i in range(pms_per_tor)])
    for row_index, row in enumerate(labels):
        for col_index, label in enumerate(row):
            if label:
                ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color="#0f172a")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("VMs on physical machine")
    return _save(fig, path)


def plot_convergence_curve(results: List[AlgorithmResult], output_dir: str = "results") -> str | None:
    best = next((result for result in results if result.name == "TA-MOACO"), None)
    history = best.metadata.get("convergence_history", []) if best else []
    if not history:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "convergence_curve.png")
    history_df = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.plot(history_df["iteration"], history_df["global_best"], marker="o", color="#246BFE", linewidth=2.5, label="Global best")
    ax.plot(history_df["iteration"], history_df["iteration_best"], marker="s", color="#16845F", linewidth=2.2, label="Iteration best")
    ax.plot(history_df["iteration"], history_df["iteration_average"], color="#C67A12", linewidth=2, alpha=0.75, label="Iteration average")
    ax.set_title("TA-MOACO Convergence Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.legend()
    history_df.to_csv(os.path.join(output_dir, "convergence_history.csv"), index=False)
    _finish_axis(ax, "Objective value")
    return _save(fig, path)


def plot_traffic_heatmap(traffic, num_vms: int, output_dir: str = "results", max_vms: int = 80) -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "traffic_heatmap.png")
    size = min(num_vms, max_vms)
    matrix = np.zeros((size, size))
    for (src, dst), demand in traffic.items():
        if src < size and dst < size:
            matrix[src, dst] = demand
            matrix[dst, src] = demand
    fig, ax = plt.subplots(figsize=(11, 9.5))
    image = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_title(f"VM Traffic Heatmap Before Placement (first {size} VMs)")
    ax.set_xlabel("Destination VM")
    ax.set_ylabel("Source VM")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Traffic demand")
    return _save(fig, path)


def plot_switch_topology_state(results: List[AlgorithmResult], data_center, traffic, output_dir: str = "results") -> str:
    _prepare(output_dir)
    path = os.path.join(output_dir, "switch_state_map.png")
    switch_rows = []
    for switch in data_center.switches.values():
        layer_order = {"core": 0, "aggregation": 1, "tor": 2}[switch.layer]
        switch_rows.append((layer_order, switch.layer, switch.id))
    switch_rows.sort()

    switch_ids = [row[2] for row in switch_rows]
    algorithm_names = [result.name for result in results]
    matrix = []
    for switch_id in switch_ids:
        row = []
        for result in results:
            active = active_switches_for_placement(data_center, result.placement, traffic)
            row.append(1 if switch_id in active else 0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, max(7, len(switch_ids) * 0.36)))
    image = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Switch Activation Matrix by Algorithm")
    ax.set_xlabel("Placement algorithm")
    ax.set_ylabel("Switches grouped by layer")
    ax.set_xticks(range(len(algorithm_names)), algorithm_names, rotation=12)
    ax.set_yticks(range(len(switch_ids)), switch_ids, fontsize=8)
    for row_index, switch_id in enumerate(switch_ids):
        layer = data_center.switches[switch_id].layer
        ax.text(-0.62, row_index, layer[:3].upper(), ha="center", va="center", fontsize=7, color="#334155")
        for col_index, value in enumerate(matrix[row_index]):
            ax.text(col_index, row_index, "ON" if value else "OFF", ha="center", va="center", fontsize=7, color="#0f172a")
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["powered down", "active"])
    return _save(fig, path)


def plot_rack_utilization(results: List[AlgorithmResult], output_dir: str = "results") -> str | None:
    """Plot average TA-MOACO utilization per rack/ToR."""

    util = _ta_moaco_pm_utilization(results)
    if util.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "rack_utilization.png")
    rack_df = util.groupby("rack_id", as_index=False)["average_utilization_pct"].mean()
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(rack_df["rack_id"].astype(str), rack_df["average_utilization_pct"], color=GREEN, width=0.7)
    _bar_labels(ax, bars, "%")
    ax.set_title("TA-MOACO Rack Utilization")
    ax.set_xlabel("Rack / ToR ID")
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, max(100, rack_df["average_utilization_pct"].max() * 1.12))
    _finish_axis(ax, "Average utilization (%)")
    return _save(fig, path)


def plot_cpu_utilization(results: List[AlgorithmResult], output_dir: str = "results") -> str | None:
    """Show per-PM CPU utilization for the recommended TA-MOACO placement."""

    util = _ta_moaco_pm_utilization(results)
    if util.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "cpu_utilization.png")
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = np.where(util["active"], BLUE, "#D8E1EC")
    ax.bar(util["pm_id"], util["cpu_utilization_pct"], color=colors, width=0.86)
    ax.axhline(80, color=ORANGE, linestyle="--", linewidth=1.6, label="80% reference")
    ax.set_title("TA-MOACO CPU Utilization by Physical Machine")
    ax.set_xlabel("Physical machine ID")
    ax.legend()
    ax.set_ylim(0, 105)
    _finish_axis(ax, "CPU utilization (%)")
    return _save(fig, path)


def plot_ram_utilization(results: List[AlgorithmResult], output_dir: str = "results") -> str | None:
    """Show per-PM RAM utilization for the recommended TA-MOACO placement."""

    util = _ta_moaco_pm_utilization(results)
    if util.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "ram_utilization.png")
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = np.where(util["active"], TEAL, "#D8E1EC")
    ax.bar(util["pm_id"], util["ram_utilization_pct"], color=colors, width=0.86)
    ax.axhline(80, color=ORANGE, linestyle="--", linewidth=1.6, label="80% reference")
    ax.set_title("TA-MOACO RAM Utilization by Physical Machine")
    ax.set_xlabel("Physical machine ID")
    ax.legend()
    ax.set_ylim(0, 105)
    _finish_axis(ax, "RAM utilization (%)")
    return _save(fig, path)


def plot_parameter_sensitivity(sensitivity_df: pd.DataFrame, output_dir: str = "results") -> str | None:
    if sensitivity_df.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "parameter_sensitivity.png")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(sensitivity_df["variant"], sensitivity_df["total_power_w"], color=BLUE)
    ax.set_title("TA-MOACO Parameter Sensitivity")
    ax.set_xlabel("Parameter variant")
    ax.set_ylabel("Total power (W)")
    plt.xticks(rotation=18)
    sensitivity_df.to_csv(os.path.join(output_dir, "parameter_sensitivity.csv"), index=False)
    _finish_axis(ax, "Total power (W)")
    return _save(fig, path)


def plot_sla_stress(stress_df: pd.DataFrame, output_dir: str = "results") -> str | None:
    if stress_df.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "sla_stress_test.png")
    fig, ax = plt.subplots(figsize=(12, 7))
    for algorithm, subset in stress_df.groupby("algorithm"):
        ax.plot(subset["load_multiplier"], subset["sla_violations"], marker="o", linewidth=2.5, label=algorithm)
    ax.set_title("SLA Stress Test")
    ax.set_xlabel("VM load multiplier")
    ax.set_ylabel("SLA violations")
    ax.legend()
    stress_df.to_csv(os.path.join(output_dir, "sla_stress_test.csv"), index=False)
    _finish_axis(ax, "SLA violations")
    return _save(fig, path)


def plot_run_history(history_df: pd.DataFrame, output_dir: str = "results") -> str | None:
    if history_df.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "run_history.png")
    recent = history_df.tail(12)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(range(1, len(recent) + 1), recent["ta_moaco_total_power_w"], marker="o", color=GREEN, label="TA-MOACO total power")
    ax.plot(range(1, len(recent) + 1), recent["ffd_total_power_w"], marker="s", color=ORANGE, label="FFD total power")
    ax.set_title("Run History Comparison")
    ax.set_xlabel("Recent run index")
    ax.set_ylabel("Total power (W)")
    ax.legend()
    _finish_axis(ax, "Total power (W)")
    return _save(fig, path)


def plot_time_window_energy(time_df: pd.DataFrame, output_dir: str = "results") -> str | None:
    if time_df is None or time_df.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "time_window_energy.png")
    fig, ax = plt.subplots(figsize=(12, 7))
    for algorithm, subset in time_df.groupby("algorithm"):
        ax.plot(subset["window"], subset["total_power_w"], marker="o", linewidth=2.5, label=algorithm)
    ax.set_title("Energy Over Real Trace Time Windows")
    ax.set_xlabel("Trace time window")
    ax.set_ylabel("Total power (W)")
    ax.legend()
    time_df.to_csv(os.path.join(output_dir, "time_window_energy.csv"), index=False)
    _finish_axis(ax, "Total power (W)")
    return _save(fig, path)


def plot_dataset_profile(trace_frame: pd.DataFrame, output_dir: str = "results") -> str | None:
    if trace_frame is None or trace_frame.empty:
        return None
    _prepare(output_dir)
    path = os.path.join(output_dir, "dataset_profile.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(trace_frame["cpu"], bins=24, color=BLUE, alpha=0.82)
    axes[0].set_title("Real Trace CPU Demand Distribution")
    axes[0].set_xlabel("CPU demand")
    axes[0].set_ylabel("VM count")
    axes[1].hist(trace_frame["ram"], bins=24, color=GREEN, alpha=0.82)
    axes[1].set_title("Real Trace RAM Demand Distribution")
    axes[1].set_xlabel("RAM demand")
    axes[1].set_ylabel("VM count")
    for ax in axes:
        _finish_axis(ax)
    return _save(fig, path)


def generate_all_charts(df: pd.DataFrame, results: List[AlgorithmResult], data_center=None, output_dir: str = "results", traffic=None, sensitivity_df=None, stress_df=None, history_df=None, trace_frame=None, time_window_df=None) -> List[str]:
    charts = [
        plot_energy_comparison(df, output_dir),
        plot_server_network_power(df, output_dir),
        plot_active_components(df, output_dir),
        plot_active_pm_comparison(df, output_dir),
        plot_active_switch_comparison(df, output_dir),
        plot_placement_map(results, output_dir),
        plot_hop_count(df, output_dir),
    ]
    convergence_path = plot_convergence_curve(results, output_dir)
    if convergence_path:
        charts.append(convergence_path)
    if data_center is not None:
        charts.append(plot_topology_placement(results, data_center, output_dir))
        for path in [
            plot_rack_utilization(results, output_dir),
            plot_cpu_utilization(results, output_dir),
            plot_ram_utilization(results, output_dir),
        ]:
            if path:
                charts.append(path)
    if traffic is not None:
        num_vms = max((max(pair) for pair in traffic.keys()), default=-1) + 1
        charts.append(plot_traffic_heatmap(traffic, num_vms, output_dir))
        if data_center is not None:
            charts.append(plot_switch_topology_state(results, data_center, traffic, output_dir))
    sensitivity_path = plot_parameter_sensitivity(sensitivity_df if sensitivity_df is not None else pd.DataFrame(), output_dir)
    if sensitivity_path:
        charts.append(sensitivity_path)
    stress_path = plot_sla_stress(stress_df if stress_df is not None else pd.DataFrame(), output_dir)
    if stress_path:
        charts.append(stress_path)
    history_path = plot_run_history(history_df if history_df is not None else pd.DataFrame(), output_dir)
    if history_path:
        charts.append(history_path)
    profile_path = plot_dataset_profile(trace_frame if trace_frame is not None else pd.DataFrame(), output_dir)
    if profile_path:
        charts.append(profile_path)
    time_window_path = plot_time_window_energy(time_window_df if time_window_df is not None else pd.DataFrame(), output_dir)
    if time_window_path:
        charts.append(time_window_path)
    return charts
