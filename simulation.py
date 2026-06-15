from __future__ import annotations

import random
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from algorithms.acs_vmp import run_acs_vmp
from algorithms.common import compute_sla_violations
from algorithms.ffd import run_ffd
from algorithms.ta_moaco import run_ta_moaco
from algorithms.ta_vmp import run_ta_vmp
from models import AlgorithmResult, DataCenter, PhysicalMachine, Placement, TrafficMatrix, VirtualMachine
from power_model import network_power, power_summary, server_power
from topology import average_traffic_weighted_hop_count, build_fat_tree_like_topology


MAX_FAT_TREE_PM_HOPS = 6.0


def generate_vms(num_vms: int, seed: int = 42) -> List[VirtualMachine]:
    """Generate realistic small-to-medium VM requests."""

    rng = np.random.default_rng(seed)
    cpu_options = np.array([4, 6, 8, 10, 12, 16, 20])
    ram_options = np.array([8, 12, 16, 24, 32, 48, 64])
    cpu_probs = np.array([0.18, 0.18, 0.22, 0.16, 0.12, 0.09, 0.05])
    ram_probs = np.array([0.18, 0.14, 0.24, 0.16, 0.16, 0.08, 0.04])
    return [
        VirtualMachine(
            id=i,
            cpu=float(rng.choice(cpu_options, p=cpu_probs)),
            ram=float(rng.choice(ram_options, p=ram_probs)),
        )
        for i in range(num_vms)
    ]


def generate_traffic_matrix(num_vms: int, seed: int = 42, groups: int = 20, profile: str = "medium") -> TrafficMatrix:
    """Create clustered traffic so locality-aware algorithms have a real signal."""

    rng = random.Random(seed)
    traffic: TrafficMatrix = {}
    group_size = max(1, num_vms // groups)
    profiles = {
        "low": {"same_probability": 0.07, "cross_probability": 0.006, "same_range": (12.0, 55.0), "cross_range": (1.0, 10.0)},
        "medium": {"same_probability": 0.12, "cross_probability": 0.012, "same_range": (25.0, 120.0), "cross_range": (2.0, 25.0)},
        "peak": {"same_probability": 0.22, "cross_probability": 0.028, "same_range": (70.0, 220.0), "cross_range": (10.0, 65.0)},
    }
    selected = profiles.get(profile, profiles["medium"])

    for i in range(num_vms):
        for j in range(i + 1, num_vms):
            same_group = (i // group_size) == (j // group_size)
            probability = selected["same_probability"] if same_group else selected["cross_probability"]
            if rng.random() < probability:
                demand_range = selected["same_range"] if same_group else selected["cross_range"]
                demand = rng.uniform(*demand_range)
                traffic[(i, j)] = round(demand, 2)
    return traffic


def load_uploaded_data(vm_csv, traffic_csv) -> Tuple[List[VirtualMachine], TrafficMatrix]:
    vm_df = pd.read_csv(vm_csv)
    required_vm_cols = {"id", "cpu", "ram"}
    if not required_vm_cols.issubset(vm_df.columns):
        raise ValueError("VM CSV must include id, cpu, and ram columns.")

    vms = [VirtualMachine(int(row.id), float(row.cpu), float(row.ram)) for row in vm_df.itertuples()]
    traffic: TrafficMatrix = {}
    if traffic_csv is not None:
        traffic_df = pd.read_csv(traffic_csv)
        required_traffic_cols = {"src_vm", "dst_vm", "demand"}
        if not required_traffic_cols.issubset(traffic_df.columns):
            raise ValueError("Traffic CSV must include src_vm, dst_vm, and demand columns.")
        for row in traffic_df.itertuples():
            src, dst = int(row.src_vm), int(row.dst_vm)
            if src != dst and float(row.demand) > 0:
                traffic[(min(src, dst), max(src, dst))] = float(row.demand)
    else:
        traffic = generate_traffic_matrix(len(vms))
    return vms, traffic


def evaluate_result(
    name: str,
    data_center: DataCenter,
    pms: List[PhysicalMachine],
    placement: Placement,
    traffic: TrafficMatrix,
    unplaced_count: int,
    baseline_total_power: float,
) -> AlgorithmResult:
    """Convert one placement into the metrics used by reports and dashboards."""

    summary = power_summary(data_center, pms, placement, traffic)
    avg_hops = average_traffic_weighted_hop_count(data_center, placement, traffic)
    localization = max(0.0, min(100.0, (1.0 - avg_hops / MAX_FAT_TREE_PM_HOPS) * 100.0))
    savings = ((baseline_total_power - summary.total_dc_power) / baseline_total_power * 100.0) if baseline_total_power else 0.0

    return AlgorithmResult(
        name=name,
        placement=placement,
        active_pms=summary.active_pms,
        active_switches=summary.active_switches,
        server_power=summary.total_server_power,
        network_power=summary.total_switch_power,
        total_power=summary.total_dc_power,
        energy_savings_pct=savings,
        average_hop_count=avg_hops,
        sla_violations=compute_sla_violations(pms, unplaced_count),
        average_utilization=summary.average_utilization,
        network_localization_score=localization,
        metadata={
            "pm_utilization": [
                {
                    "pm_id": pm.id,
                    "rack_id": pm.rack_id,
                    "cpu_utilization_pct": round(pm.cpu_utilization * 100.0, 4),
                    "ram_utilization_pct": round(pm.ram_utilization * 100.0, 4),
                    "average_utilization_pct": round(pm.utilization * 100.0, 4),
                    "vm_count": len(pm.vm_ids),
                    "active": pm.is_active,
                }
                for pm in pms
            ]
        },
    )


def best_algorithm_name(results: List[AlgorithmResult]) -> str:
    """Select the best algorithm, preferring no SLA violations then low power."""

    if not results:
        return ""
    return min(results, key=lambda result: (result.sla_violations > 0, result.total_power, result.average_hop_count)).name


def results_to_dataframe(results: List[AlgorithmResult]) -> pd.DataFrame:
    best_name = best_algorithm_name(results)
    return pd.DataFrame(
        [
            {
                "best_algorithm": "YES" if result.name == best_name else "",
                "algorithm": result.name,
                "active_pms": result.active_pms,
                "active_switches": result.active_switches,
                "server_power_w": round(result.server_power, 2),
                "network_power_w": round(result.network_power, 2),
                "switch_power_w": round(result.network_power, 2),
                "total_power_w": round(result.total_power, 2),
                "total_dc_power_w": round(result.total_power, 2),
                "energy_savings_pct": round(result.energy_savings_pct, 2),
                "average_hop_count": round(result.average_hop_count, 2),
                "network_localization_score": round(result.network_localization_score, 2),
                "average_utilization_pct": round(result.average_utilization * 100.0, 2),
                "sla_violations": result.sla_violations,
            }
            for result in results
        ]
    )


def run_simulation(
    num_pms: int = 100,
    num_vms: int = 400,
    seed: int = 42,
    ants: int = 50,
    iterations: int = 25,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.1,
    vms: List[VirtualMachine] | None = None,
    traffic: TrafficMatrix | None = None,
    traffic_profile: str = "medium",
) -> Tuple[pd.DataFrame, List[AlgorithmResult], DataCenter]:
    """Run all algorithms and return presentation-ready metrics."""

    data_center = build_fat_tree_like_topology(num_pms=num_pms)
    vms = vms or generate_vms(num_vms, seed=seed)
    traffic = traffic or generate_traffic_matrix(len(vms), seed=seed, profile=traffic_profile)

    ffd_placement, ffd_pms, ffd_unplaced = run_ffd(data_center.pms, vms)
    ffd_server = server_power(ffd_pms)
    ffd_network = network_power(data_center, ffd_placement, traffic)
    baseline_total = ffd_server + ffd_network

    raw_results = [
        ("FFD", ffd_placement, ffd_pms, ffd_unplaced),
    ]

    acs_placement, acs_pms, acs_unplaced = run_acs_vmp(
        data_center.pms, vms, ants=ants, iterations=max(8, iterations // 2), seed=seed + 1
    )
    raw_results.append(("ACS-VMP", acs_placement, acs_pms, acs_unplaced))

    ta_placement, ta_pms, ta_unplaced = run_ta_vmp(data_center.pms, vms, traffic, data_center)
    raw_results.append(("Traffic-Aware VMP", ta_placement, ta_pms, ta_unplaced))

    moaco_placement, moaco_pms, moaco_unplaced, convergence_history = run_ta_moaco(
        data_center.pms,
        vms,
        traffic,
        data_center,
        ants=ants,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
        evaporation=evaporation,
        seed=seed + 2,
        return_history=True,
    )
    raw_results.append(("TA-MOACO", moaco_placement, moaco_pms, moaco_unplaced))

    results = [
        evaluate_result(name, data_center, pms, placement, traffic, unplaced, baseline_total)
        for name, placement, pms, unplaced in raw_results
    ]
    for result in results:
        result.metadata["traffic_profile"] = traffic_profile
    for result in results:
        if result.name == "TA-MOACO":
            result.metadata["convergence_history"] = convergence_history
            result.metadata["ta_moaco_parameters"] = {
                "ants": ants,
                "iterations": iterations,
                "alpha": alpha,
                "beta": beta,
                "evaporation": evaporation,
            }
    return results_to_dataframe(results), results, data_center


def save_results_csv(df: pd.DataFrame, output_dir: str = "results") -> str:
    """Save both legacy results.csv and the presentation comparison CSV."""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.csv")
    comparison_path = os.path.join(output_dir, "algorithm_comparison.csv")
    df.to_csv(path, index=False)
    df.to_csv(comparison_path, index=False)
    return path
