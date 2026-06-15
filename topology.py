from __future__ import annotations

import math
from typing import Dict, Iterable, List, Set

from models import DataCenter, PhysicalMachine, Placement, Switch, TrafficMatrix


SWITCH_POWER = {
    "tor": {"chassis": 45.0, "port": 2.0},
    "aggregation": {"chassis": 65.0, "port": 3.0},
    "core": {"chassis": 95.0, "port": 4.0},
}


def build_fat_tree_like_topology(num_pms: int = 100, pms_per_tor: int = 10) -> DataCenter:
    """Create a compact fat-tree style topology.

    The layout has ToR/edge switches under aggregation switches under core
    switches. It is intentionally simple enough for classroom explanation while
    still letting traffic locality change active switches and hop counts.
    """

    num_tors = math.ceil(num_pms / pms_per_tor)
    tors_per_pod = 4
    aggs_per_pod = 2
    num_core = 4
    num_pods = math.ceil(num_tors / tors_per_pod)

    switches: Dict[str, Switch] = {}
    pm_to_tor: Dict[int, str] = {}
    tor_to_agg: Dict[str, List[str]] = {}
    agg_to_core: Dict[str, List[str]] = {}
    pods: Dict[int, List[str]] = {}
    racks: Dict[int, List[int]] = {}

    for core_id in range(num_core):
        switches[f"core-{core_id}"] = Switch(
            f"core-{core_id}",
            "core",
            None,
            port_count=num_pods * aggs_per_pod,
            chassis_power=SWITCH_POWER["core"]["chassis"],
            port_power=SWITCH_POWER["core"]["port"],
        )

    pms: List[PhysicalMachine] = []
    for tor_index in range(num_tors):
        pod_id = tor_index // tors_per_pod
        rack_id = tor_index
        tor_id = f"tor-{tor_index}"
        pods.setdefault(pod_id, [])
        racks.setdefault(rack_id, [])
        switches[tor_id] = Switch(
            tor_id,
            "tor",
            pod_id,
            port_count=pms_per_tor + aggs_per_pod,
            rack_id=rack_id,
            chassis_power=SWITCH_POWER["tor"]["chassis"],
            port_power=SWITCH_POWER["tor"]["port"],
        )
        pods[pod_id].append(tor_id)

        agg_ids = []
        for agg_offset in range(aggs_per_pod):
            agg_id = f"agg-{pod_id}-{agg_offset}"
            if agg_id not in switches:
                switches[agg_id] = Switch(
                    agg_id,
                    "aggregation",
                    pod_id,
                    port_count=tors_per_pod + num_core,
                    chassis_power=SWITCH_POWER["aggregation"]["chassis"],
                    port_power=SWITCH_POWER["aggregation"]["port"],
                )
                agg_to_core[agg_id] = [f"core-{core_id}" for core_id in range(num_core)]
                pods[pod_id].append(agg_id)
            agg_ids.append(agg_id)
        tor_to_agg[tor_id] = agg_ids

        for slot in range(pms_per_tor):
            pm_id = tor_index * pms_per_tor + slot
            if pm_id >= num_pms:
                break
            pm_to_tor[pm_id] = tor_id
            pms.append(
                PhysicalMachine(
                    id=pm_id,
                    cpu_capacity=100.0,
                    ram_capacity=256.0,
                    tor_id=tor_index,
                    pod_id=pod_id,
                    rack_id=rack_id,
                )
            )
            racks[rack_id].append(pm_id)

    return DataCenter(
        pms=pms,
        switches=switches,
        pm_to_tor=pm_to_tor,
        tor_to_agg=tor_to_agg,
        agg_to_core=agg_to_core,
        pods=pods,
        racks=racks,
    )


def hop_count_between_pms(data_center: DataCenter, pm_a: int, pm_b: int) -> int:
    """Return PM-to-PM hop count including server uplink/downlink edges."""

    path = path_switches_between_pms(data_center, pm_a, pm_b)
    return 0 if not path else len(path) + 1


def path_switches_between_pms(data_center: DataCenter, pm_a: int, pm_b: int) -> List[str]:
    """Return the switch path used between two PMs.

    Same host returns an empty path. Same rack/ToR uses the ToR only. Same pod
    uses ToR -> aggregation -> ToR. Cross-pod uses ToR -> aggregation -> core
    -> aggregation -> ToR.
    """

    if pm_a == pm_b:
        return []
    tor_a = data_center.pm_to_tor[pm_a]
    tor_b = data_center.pm_to_tor[pm_b]
    if tor_a == tor_b:
        return [tor_a]
    src_pod = data_center.switches[tor_a].pod_id
    dst_pod = data_center.switches[tor_b].pod_id
    src_agg = data_center.tor_to_agg[tor_a][0]
    if src_pod == dst_pod:
        return [tor_a, src_agg, tor_b]
    dst_agg = data_center.tor_to_agg[tor_b][0]
    core = data_center.agg_to_core[src_agg][0]
    return [tor_a, src_agg, core, dst_agg, tor_b]


def active_switches_for_placement(data_center: DataCenter, placement: Placement, traffic: TrafficMatrix) -> Set[str]:
    active: Set[str] = set()
    active_pms = set(placement.values())

    for pm_id in active_pms:
        active.add(data_center.pm_to_tor[pm_id])

    # Switches become active only if they are needed by traffic paths.
    for (src_vm, dst_vm), demand in traffic.items():
        if demand <= 0 or src_vm not in placement or dst_vm not in placement:
            continue
        src_pm = placement[src_vm]
        dst_pm = placement[dst_vm]
        if src_pm == dst_pm:
            continue
        active.update(path_switches_between_pms(data_center, src_pm, dst_pm))
    return active


def active_ports_by_switch(data_center: DataCenter, active_switches: Iterable[str], placement: Placement, traffic: TrafficMatrix) -> Dict[str, int]:
    ports = {switch_id: 0 for switch_id in active_switches}

    for pm_id in set(placement.values()):
        tor = data_center.pm_to_tor[pm_id]
        if tor in ports:
            ports[tor] += 1

    for (src_vm, dst_vm), demand in traffic.items():
        if demand <= 0 or src_vm not in placement or dst_vm not in placement:
            continue
        src_pm = placement[src_vm]
        dst_pm = placement[dst_vm]
        if src_pm == dst_pm:
            continue
        path = path_switches_between_pms(data_center, src_pm, dst_pm)
        for switch_id in path:
            ports[switch_id] = ports.get(switch_id, 0) + 2

    return {switch_id: min(count, data_center.switches[switch_id].port_count) for switch_id, count in ports.items()}


def switch_runtime_state(data_center: DataCenter, placement: Placement, traffic: TrafficMatrix) -> Dict[str, Switch]:
    """Return switch metadata with active port and sleep-state values filled in."""

    active = active_switches_for_placement(data_center, placement, traffic)
    ports = active_ports_by_switch(data_center, active, placement, traffic)
    states: Dict[str, Switch] = {}
    for switch_id, switch in data_center.switches.items():
        active_ports = ports.get(switch_id, 0)
        states[switch_id] = Switch(
            id=switch.id,
            layer=switch.layer,
            pod_id=switch.pod_id,
            port_count=switch.port_count,
            rack_id=switch.rack_id,
            chassis_power=switch.chassis_power,
            port_power=switch.port_power,
            active_ports=active_ports,
            sleep_state=switch_id not in active,
        )
    return states


def average_traffic_weighted_hop_count(data_center: DataCenter, placement: Placement, traffic: TrafficMatrix) -> float:
    weighted_hops = 0.0
    total_traffic = 0.0
    for (src_vm, dst_vm), demand in traffic.items():
        if src_vm in placement and dst_vm in placement and demand > 0:
            weighted_hops += demand * hop_count_between_pms(data_center, placement[src_vm], placement[dst_vm])
            total_traffic += demand
    return weighted_hops / total_traffic if total_traffic else 0.0
