from __future__ import annotations

from typing import List, Tuple

from algorithms.common import network_blind_host_order, sorted_vms_by_demand
from models import PhysicalMachine, Placement, VirtualMachine, clone_pms


def run_ffd(base_pms: List[PhysicalMachine], vms: List[VirtualMachine]) -> Tuple[Placement, List[PhysicalMachine], int]:
    """First Fit Decreasing: simple consolidation baseline."""

    pms = clone_pms(base_pms)
    placement: Placement = {}
    unplaced = 0
    host_order = network_blind_host_order(pms)

    for vm in sorted_vms_by_demand(vms):
        placed = False
        for pm_id in host_order:
            if pms[pm_id].place(vm):
                placement[vm.id] = pm_id
                placed = True
                break
        if not placed:
            unplaced += 1

    return placement, pms, unplaced
