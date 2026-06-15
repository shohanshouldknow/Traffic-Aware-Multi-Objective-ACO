from __future__ import annotations

import random
from typing import Dict, List, Tuple

from models import PhysicalMachine, Placement, TrafficMatrix, VirtualMachine
from topology import hop_count_between_pms


def sorted_vms_by_demand(vms: List[VirtualMachine]) -> List[VirtualMachine]:
    return sorted(vms, key=lambda vm: (vm.cpu + vm.ram / 4.0), reverse=True)


def sorted_vms_by_traffic_and_demand(vms: List[VirtualMachine], traffic: TrafficMatrix) -> List[VirtualMachine]:
    traffic_totals: Dict[int, float] = {vm.id: 0.0 for vm in vms}
    for (src, dst), demand in traffic.items():
        traffic_totals[src] = traffic_totals.get(src, 0.0) + demand
        traffic_totals[dst] = traffic_totals.get(dst, 0.0) + demand
    return sorted(vms, key=lambda vm: (traffic_totals.get(vm.id, 0.0), vm.cpu + vm.ram / 4.0), reverse=True)


def network_blind_host_order(pms: List[PhysicalMachine]) -> List[int]:
    """A deterministic but topology-agnostic host order for baselines.

    It prevents FFD from receiving an accidental advantage from PM IDs being
    grouped by ToR in the synthetic topology.
    """

    if not pms:
        return []
    return [pm.id for pm in sorted(pms, key=lambda pm: (pm.id % 20, pm.id // 20))]


def feasible_pm_ids(pms: List[PhysicalMachine], vm: VirtualMachine) -> List[int]:
    """Return IDs of PMs with enough remaining CPU and RAM for the VM."""

    return [pm.id for pm in pms if pm.can_host(vm)]


def compact_candidate_pm_ids(pms: List[PhysicalMachine], vm: VirtualMachine, host_order: List[int]) -> List[int]:
    active = [pm_id for pm_id in host_order if pms[pm_id].is_active and pms[pm_id].can_host(vm)]
    if active:
        return active
    for pm_id in host_order:
        if pms[pm_id].can_host(vm):
            return [pm_id]
    return []


def resource_score(pm: PhysicalMachine, vm: VirtualMachine) -> float:
    """Favor hosts that become well packed without exceeding capacity."""

    cpu_after = (pm.used_cpu + vm.cpu) / pm.cpu_capacity
    ram_after = (pm.used_ram + vm.ram) / pm.ram_capacity
    balance_penalty = abs(cpu_after - ram_after)
    packing_bonus = (cpu_after + ram_after) / 2.0
    return max(0.01, 0.65 * packing_bonus + 0.35 * (1.0 - balance_penalty))


TrafficAdjacency = Dict[int, List[Tuple[int, float]]]


def build_traffic_adjacency(traffic: TrafficMatrix) -> TrafficAdjacency:
    adjacency: TrafficAdjacency = {}
    for (src, dst), demand in traffic.items():
        adjacency.setdefault(src, []).append((dst, demand))
        adjacency.setdefault(dst, []).append((src, demand))
    return adjacency


def traffic_affinity_score(data_center, placement: Placement, traffic: TrafficMatrix | TrafficAdjacency, candidate_pm: int, vm_id: int) -> float:
    """Higher when communicating VMs are already near the candidate PM."""

    score = 0.0
    if not traffic:
        neighbors = []
    else:
        first_key = next(iter(traffic.keys()))
        if isinstance(first_key, tuple):
            adjacency = build_traffic_adjacency(traffic)  # Used by simple callers if they pass the raw matrix.
            neighbors = adjacency.get(vm_id, [])
        else:
            neighbors = traffic.get(vm_id, [])

    if neighbors is None:
        adjacency = build_traffic_adjacency(traffic)  # Used by simple greedy callers if they pass the raw matrix.
        neighbors = adjacency.get(vm_id, [])

    for other, demand in neighbors:
        if demand <= 0:
            continue
        if other not in placement:
            continue
        other_pm = placement[other]
        hop = hop_count_between_pms(data_center, candidate_pm, other_pm)
        score += demand / (1.0 + hop)
    return 1.0 + score


def roulette_choice(weighted_items: List[Tuple[int, float]], rng: random.Random) -> int:
    total = sum(max(0.0, weight) for _, weight in weighted_items)
    if total <= 0:
        return rng.choice([item for item, _ in weighted_items])
    pick = rng.random() * total
    cumulative = 0.0
    for item, weight in weighted_items:
        cumulative += max(0.0, weight)
        if cumulative >= pick:
            return item
    return weighted_items[-1][0]


def compute_sla_violations(pms: List[PhysicalMachine], unplaced_count: int) -> int:
    overloads = sum(1 for pm in pms if pm.used_cpu > pm.cpu_capacity + 1e-9 or pm.used_ram > pm.ram_capacity + 1e-9)
    return overloads + unplaced_count
