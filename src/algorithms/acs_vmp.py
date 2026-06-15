from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from algorithms.common import compact_candidate_pm_ids, network_blind_host_order, resource_score, roulette_choice, sorted_vms_by_demand
from models import PhysicalMachine, Placement, VirtualMachine, clone_pms


def run_acs_vmp(
    base_pms: List[PhysicalMachine],
    vms: List[VirtualMachine],
    ants: int = 50,
    iterations: int = 20,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.1,
    seed: int = 7,
) -> Tuple[Placement, List[PhysicalMachine], int]:
    """Server-centric ACS-VMP baseline.

    This version ignores VM traffic and tries to minimize active hosts by using
    pheromone and resource packing only.
    """

    rng = random.Random(seed)
    ordered_vms = sorted_vms_by_demand(vms)
    host_order = network_blind_host_order(base_pms)
    pheromone = np.ones((len(vms), len(base_pms)), dtype=float)

    best_placement: Placement = {}
    best_pms = clone_pms(base_pms)
    best_score = float("inf")
    best_unplaced = len(vms)

    for _ in range(iterations):
        ant_solutions = []
        for _ant in range(ants):
            pms = clone_pms(base_pms)
            placement: Placement = {}
            unplaced = 0

            for vm in ordered_vms:
                candidates = compact_candidate_pm_ids(pms, vm, host_order)
                if not candidates:
                    unplaced += 1
                    continue
                weighted = []
                for pm_id in candidates:
                    score = (pheromone[vm.id, pm_id] ** alpha) * (resource_score(pms[pm_id], vm) ** beta)
                    weighted.append((pm_id, score))
                chosen_pm = roulette_choice(weighted, rng)
                pms[chosen_pm].place(vm)
                placement[vm.id] = chosen_pm

            active_pms = sum(1 for pm in pms if pm.is_active)
            score = active_pms * 1000 + unplaced * 100000 + sum(abs(pm.cpu_utilization - pm.ram_utilization) for pm in pms)
            ant_solutions.append((score, placement, pms, unplaced))
            if score < best_score:
                best_score, best_placement, best_pms, best_unplaced = score, placement, pms, unplaced

        pheromone *= 1.0 - evaporation
        for rank, (_score, placement, _pms, _unplaced) in enumerate(sorted(ant_solutions, key=lambda x: x[0])[:5], start=1):
            deposit = 1.0 / rank
            for vm_id, pm_id in placement.items():
                pheromone[vm_id, pm_id] += deposit

    return best_placement, best_pms, best_unplaced
