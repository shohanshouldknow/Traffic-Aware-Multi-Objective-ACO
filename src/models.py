from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class VirtualMachine:
    """A VM request with CPU and RAM demand."""

    id: int
    cpu: float
    ram: float


@dataclass
class PhysicalMachine:
    """A physical host with finite capacity and a list of assigned VMs."""

    id: int
    cpu_capacity: float
    ram_capacity: float
    tor_id: int
    pod_id: int
    rack_id: int = 0
    state: str = "sleep"
    vm_ids: List[int] = field(default_factory=list)
    used_cpu: float = 0.0
    used_ram: float = 0.0

    def can_host(self, vm: VirtualMachine) -> bool:
        return (
            self.used_cpu + vm.cpu <= self.cpu_capacity + 1e-9
            and self.used_ram + vm.ram <= self.ram_capacity + 1e-9
        )

    def place(self, vm: VirtualMachine) -> bool:
        if not self.can_host(vm):
            return False
        self.vm_ids.append(vm.id)
        self.used_cpu += vm.cpu
        self.used_ram += vm.ram
        self.state = "active"
        return True

    @property
    def cpu_utilization(self) -> float:
        return min(1.0, self.used_cpu / self.cpu_capacity)

    @property
    def ram_utilization(self) -> float:
        return min(1.0, self.used_ram / self.ram_capacity)

    @property
    def is_active(self) -> bool:
        return self.state == "active" or bool(self.vm_ids)

    @property
    def utilization(self) -> float:
        """Average CPU/RAM utilization used for infrastructure reporting."""

        return (self.cpu_utilization + self.ram_utilization) / 2.0

    def clone_empty(self) -> "PhysicalMachine":
        return PhysicalMachine(
            id=self.id,
            cpu_capacity=self.cpu_capacity,
            ram_capacity=self.ram_capacity,
            tor_id=self.tor_id,
            pod_id=self.pod_id,
            rack_id=self.rack_id,
            state="sleep",
        )


@dataclass(frozen=True)
class Switch:
    """Network switch metadata used by the power model."""

    id: str
    layer: str
    pod_id: Optional[int]
    port_count: int
    rack_id: Optional[int] = None
    chassis_power: float = 0.0
    port_power: float = 0.0
    active_ports: int = 0
    sleep_state: bool = True


@dataclass
class DataCenter:
    pms: List[PhysicalMachine]
    switches: Dict[str, Switch]
    pm_to_tor: Dict[int, str]
    tor_to_agg: Dict[str, List[str]]
    agg_to_core: Dict[str, List[str]]
    pods: Dict[int, List[str]] = field(default_factory=dict)
    racks: Dict[int, List[int]] = field(default_factory=dict)


Placement = Dict[int, int]
TrafficMatrix = Dict[Tuple[int, int], float]


@dataclass
class AlgorithmResult:
    name: str
    placement: Placement
    active_pms: int
    active_switches: int
    server_power: float
    network_power: float
    total_power: float
    energy_savings_pct: float
    average_hop_count: float
    sla_violations: int
    average_utilization: float = 0.0
    network_localization_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def clone_pms(pms: List[PhysicalMachine]) -> List[PhysicalMachine]:
    return [pm.clone_empty() for pm in pms]
