from __future__ import annotations

from typing import List, Tuple

from algorithms.common import build_traffic_adjacency, resource_score, sorted_vms_by_demand, traffic_affinity_score
from models import PhysicalMachine, Placement, TrafficMatrix, VirtualMachine, clone_pms


def run_ta_vmp(base_pms: List[PhysicalMachine], vms: List[VirtualMachine], traffic: TrafficMatrix, data_center) -> Tuple[Placement, List[PhysicalMachine], int]:
    """Greedy traffic-aware placement baseline."""

    pms = clone_pms(base_pms)
    placement: Placement = {}
    unplaced = 0
    traffic_adjacency = build_traffic_adjacency(traffic)

    for vm in sorted_vms_by_demand(vms):
        best_pm = None
        best_score = -1.0
        for pm in pms:
            if not pm.can_host(vm):
                continue
            locality = traffic_affinity_score(data_center, placement, traffic_adjacency, pm.id, vm.id)
            packing = resource_score(pm, vm)
            # This baseline is traffic-aware but not multi-objective: it is
            # willing to open extra servers when that improves locality.
            active_bias = 0.15 if pm.is_active else 0.28
            score = locality * 1.15 + packing * active_bias
            if score > best_score:
                best_pm = pm
                best_score = score
        if best_pm is None:
            unplaced += 1
        else:
            best_pm.place(vm)
            placement[vm.id] = best_pm.id

    return placement, pms, unplaced
