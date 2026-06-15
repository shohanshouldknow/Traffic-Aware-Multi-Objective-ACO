from __future__ import annotations

from dataclasses import dataclass
from typing import List

from models import PhysicalMachine, Placement, TrafficMatrix
from topology import switch_runtime_state


P_IDLE = 120.0
P_MAX = 250.0
SWITCH_CHASSIS_POWER = {
    "tor": 45.0,
    "aggregation": 65.0,
    "core": 95.0,
}
ACTIVE_PORT_POWER = {
    "tor": 2.0,
    "aggregation": 3.0,
    "core": 4.0,
}


@dataclass(frozen=True)
class PowerSummary:
    """Validated data-center power and utilization summary."""

    total_server_power: float
    total_switch_power: float
    total_dc_power: float
    active_pms: int
    active_switches: int
    average_utilization: float


def server_power(pms: List[PhysicalMachine]) -> float:
    """Linear server power model from the paper requirements."""

    total = 0.0
    for pm in pms:
        if pm.is_active:
            total += P_IDLE + (P_MAX - P_IDLE) * pm.cpu_utilization
    return total


def network_power(data_center, placement: Placement, traffic: TrafficMatrix) -> float:
    """Switch power: chassis + active_ports * port_power."""

    total = 0.0
    for switch in switch_runtime_state(data_center, placement, traffic).values():
        if switch.sleep_state:
            continue
        chassis_power = switch.chassis_power or SWITCH_CHASSIS_POWER[switch.layer]
        port_power = switch.port_power or ACTIVE_PORT_POWER[switch.layer]
        total += chassis_power + port_power * switch.active_ports
    return total


def average_pm_utilization(pms: List[PhysicalMachine]) -> float:
    """Average CPU utilization across active PMs.

    The server power formula is CPU-utilization based, so this metric uses the
    same utilization basis for consistency with reported power.
    """

    active = [pm for pm in pms if pm.is_active]
    if not active:
        return 0.0
    return sum(pm.cpu_utilization for pm in active) / len(active)


def power_summary(data_center, pms: List[PhysicalMachine], placement: Placement, traffic: TrafficMatrix) -> PowerSummary:
    """Calculate server, switch, total data-center power, and active components."""

    total_server = server_power(pms)
    switch_states = switch_runtime_state(data_center, placement, traffic)
    active_switches = [switch for switch in switch_states.values() if not switch.sleep_state]
    total_switch = 0.0
    for switch in active_switches:
        chassis_power = switch.chassis_power or SWITCH_CHASSIS_POWER[switch.layer]
        port_power = switch.port_power or ACTIVE_PORT_POWER[switch.layer]
        total_switch += chassis_power + switch.active_ports * port_power
    return PowerSummary(
        total_server_power=total_server,
        total_switch_power=total_switch,
        total_dc_power=total_server + total_switch,
        active_pms=sum(1 for pm in pms if pm.is_active),
        active_switches=len(active_switches),
        average_utilization=average_pm_utilization(pms),
    )
