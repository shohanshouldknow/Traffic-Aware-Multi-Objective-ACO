from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

from algorithms.common import build_traffic_adjacency, feasible_pm_ids, resource_score, roulette_choice, sorted_vms_by_traffic_and_demand, traffic_affinity_score
from models import PhysicalMachine, Placement, TrafficMatrix, VirtualMachine, clone_pms
from power_model import network_power, server_power
from topology import average_traffic_weighted_hop_count


def _normalize(values: Dict[int, float]) -> Dict[int, float]:
    """Scale candidate scores into a stable 0-1 range for ACO weighting."""

    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if abs(high - low) < 1e-12:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _candidate_pool(
    pms: List[PhysicalMachine],
    vm: VirtualMachine,
    placement: Placement,
    traffic_adjacency,
    data_center,
    active_limit: int = 18,
    inactive_limit: int = 6,
) -> List[int]:
    """Return a compact list of good PM candidates to keep each ant fast.

    Active PMs are preferred because consolidation turns idle servers and
    switches off. A few inactive PMs remain available so ants can still escape
    poor early packing choices when active servers are not a good fit.
    """

    feasible = feasible_pm_ids(pms, vm)
    active = [pm_id for pm_id in feasible if pms[pm_id].is_active]
    inactive = [pm_id for pm_id in feasible if not pms[pm_id].is_active]
    if not active:
        return inactive[: max(1, inactive_limit)]

    def candidate_score(pm_id: int) -> float:
        return (
            0.55 * resource_score(pms[pm_id], vm)
            + 0.45 * traffic_affinity_score(data_center, placement, traffic_adjacency, pm_id, vm.id)
        )

    selected_active = sorted(active, key=candidate_score, reverse=True)[:active_limit]
    selected_inactive = sorted(inactive, key=candidate_score, reverse=True)[:inactive_limit]
    return selected_active + selected_inactive


def _heuristic_scores(
    pms: List[PhysicalMachine],
    vm: VirtualMachine,
    candidates: List[int],
    placement: Placement,
    traffic_adjacency,
    data_center,
) -> Dict[int, float]:
    """Combine resource packing and traffic locality into one heuristic value."""

    resource_values = {pm_id: resource_score(pms[pm_id], vm) for pm_id in candidates}
    affinity_values = {
        pm_id: traffic_affinity_score(data_center, placement, traffic_adjacency, pm_id, vm.id)
        for pm_id in candidates
    }
    normalized_resource = _normalize(resource_values)
    normalized_affinity = _normalize(affinity_values)

    scores: Dict[int, float] = {}
    for pm_id in candidates:
        pm = pms[pm_id]
        cpu_after = (pm.used_cpu + vm.cpu) / pm.cpu_capacity
        ram_after = (pm.used_ram + vm.ram) / pm.ram_capacity
        consolidation = min(1.0, (cpu_after + ram_after) / 2.0)
        active_bonus = 1.18 if pm.is_active else 0.72

        # Resource keeps placements feasible and compact; affinity pulls chatty
        # VMs closer together so fewer switches and network hops are needed.
        heuristic = (
            0.36 * normalized_resource[pm_id]
            + 0.46 * normalized_affinity[pm_id]
            + 0.18 * consolidation
        ) * active_bonus
        scores[pm_id] = max(0.01, heuristic)
    return scores


def _objective(data_center, placement: Placement, pms: List[PhysicalMachine], traffic: TrafficMatrix, unplaced: int) -> Tuple[float, float, float, int]:
    """Evaluate the multi-objective cost minimized by TA-MOACO."""

    s_power = server_power(pms)
    n_power = network_power(data_center, placement, traffic)
    avg_hops = average_traffic_weighted_hop_count(data_center, placement, traffic)
    active_pms = sum(1 for pm in pms if pm.is_active)
    score = s_power + n_power + 120.0 * avg_hops + 40.0 * active_pms + 100000.0 * unplaced
    return score, s_power + n_power, avg_hops, active_pms


def run_ta_moaco(
    base_pms: List[PhysicalMachine],
    vms: List[VirtualMachine],
    traffic: TrafficMatrix,
    data_center,
    ants: int = 50,
    iterations: int = 25,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.1,
    seed: int = 11,
    return_history: bool = False,
) -> Tuple[Placement, List[PhysicalMachine], int] | Tuple[Placement, List[PhysicalMachine], int, List[dict]]:
    """Traffic-Aware Multi-Objective ACO for VM placement.

    Probability(vm -> pm) is proportional to:
        pheromone^alpha * (resource_heuristic * traffic_affinity)^beta

    The update rewards ants that reduce total power, active PMs, and average
    hop count, which is the paper's central idea.
    """

    rng = random.Random(seed)
    ants = max(1, int(ants))
    iterations = max(1, int(iterations))
    alpha = max(0.0, float(alpha))
    beta = max(0.0, float(beta))
    evaporation = min(0.9, max(0.01, float(evaporation)))

    ordered_vms = sorted_vms_by_traffic_and_demand(vms, traffic)
    vm_to_row = {vm.id: row for row, vm in enumerate(vms)}
    pheromone = np.ones((len(vms), len(base_pms)), dtype=float)
    initial_pheromone = 1.0
    min_pheromone = 0.05
    max_pheromone = 8.0
    local_decay = 0.04
    traffic_adjacency = build_traffic_adjacency(traffic)

    best_placement: Placement = {}
    best_pms = clone_pms(base_pms)
    best_unplaced = len(vms)
    best_score = float("inf")
    best_total_power = float("inf")
    best_avg_hops = 0.0
    best_active_pms = 0
    stagnation = 0
    convergence_history: List[dict] = []

    for iteration in range(iterations):
        ant_solutions = []
        progress = iteration / max(1, iterations - 1)
        exploitation_rate = min(0.9, 0.65 + 0.25 * progress)
        exploration_rate = max(0.04, 0.22 * (1.0 - progress))

        for ant_index in range(ants):
            pms = clone_pms(base_pms)
            placement: Placement = {}
            unplaced = 0

            for vm in ordered_vms:
                candidates = _candidate_pool(pms, vm, placement, traffic_adjacency, data_center)
                if not candidates:
                    unplaced += 1
                    continue

                heuristic = _heuristic_scores(pms, vm, candidates, placement, traffic_adjacency, data_center)
                vm_row = vm_to_row[vm.id]
                weighted = [
                    (pm_id, (pheromone[vm_row, pm_id] ** alpha) * (heuristic[pm_id] ** beta))
                    for pm_id in candidates
                ]

                # Early iterations deliberately explore; later iterations
                # exploit the strongest pheromone-plus-heuristic candidate.
                if rng.random() < exploration_rate:
                    top_window = sorted(weighted, key=lambda item: item[1], reverse=True)[: max(2, min(6, len(weighted)))]
                    chosen_pm = rng.choice([pm_id for pm_id, _weight in top_window])
                elif rng.random() < exploitation_rate:
                    chosen_pm = max(weighted, key=lambda item: item[1])[0]
                else:
                    chosen_pm = roulette_choice(weighted, rng)

                pms[chosen_pm].place(vm)
                placement[vm.id] = chosen_pm
                pheromone[vm_row, chosen_pm] = (
                    (1.0 - local_decay) * pheromone[vm_row, chosen_pm]
                    + local_decay * initial_pheromone
                )

            objective, total_power, avg_hops, active_pms = _objective(data_center, placement, pms, traffic, unplaced)
            ant_solutions.append((objective, total_power, avg_hops, active_pms, placement, pms, unplaced, ant_index))

            if objective < best_score:
                previous_best = best_score
                best_score = objective
                best_total_power = total_power
                best_avg_hops = avg_hops
                best_active_pms = active_pms
                best_placement = placement
                best_pms = pms
                best_unplaced = unplaced
                stagnation = 0 if previous_best == float("inf") else stagnation

        previous_global_best = convergence_history[-1]["global_best"] if convergence_history else float("inf")

        # Global evaporation prevents stale early choices from dominating.
        pheromone *= 1.0 - evaporation
        elite_count = max(2, min(10, ants // 5))
        elite = sorted(ant_solutions, key=lambda x: x[0])[:elite_count]
        iteration_best = elite[0][0] if elite else best_score
        iteration_average = sum(solution[0] for solution in ant_solutions) / len(ant_solutions)
        iteration_best_power = elite[0][1] if elite else best_total_power
        iteration_best_hops = elite[0][2] if elite else best_avg_hops

        if best_score >= previous_global_best - 1e-9:
            stagnation += 1
        else:
            stagnation = 0

        # Elite ants reinforce their placements; the global best also deposits
        # pheromone so convergence is visible across iterations.
        for rank, (objective, _total_power, _avg_hops, _active_pms, placement, _pms, _unplaced, _ant_index) in enumerate(elite, start=1):
            quality = best_score / max(objective, 1.0)
            deposit = (1.0 / rank) * max(0.05, quality)
            for vm_id, pm_id in placement.items():
                pheromone[vm_to_row[vm_id], pm_id] += deposit

        global_deposit = 1.25
        for vm_id, pm_id in best_placement.items():
            pheromone[vm_to_row[vm_id], pm_id] += global_deposit

        if stagnation >= 5:
            # Mild smoothing re-opens search if all ants keep finding the same
            # quality for several rounds without discarding learned structure.
            pheromone = 0.85 * pheromone + 0.15 * initial_pheromone
            stagnation = 0

        np.clip(pheromone, min_pheromone, max_pheromone, out=pheromone)
        convergence_history.append(
            {
                "iteration": iteration + 1,
                "iteration_best": round(iteration_best, 4),
                "iteration_best_power_w": round(iteration_best_power, 4),
                "iteration_best_hops": round(iteration_best_hops, 4),
                "global_best": round(best_score, 4),
                "global_best_power_w": round(best_total_power, 4),
                "global_best_hops": round(best_avg_hops, 4),
                "global_best_active_pms": best_active_pms,
                "iteration_average": round(iteration_average, 4),
                "exploration_rate": round(exploration_rate, 4),
                "exploitation_rate": round(exploitation_rate, 4),
            }
        )

    if return_history:
        return best_placement, best_pms, best_unplaced, convergence_history
    return best_placement, best_pms, best_unplaced
