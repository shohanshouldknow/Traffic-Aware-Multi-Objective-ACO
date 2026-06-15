from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models import VirtualMachine
from power_model import P_IDLE, P_MAX, average_pm_utilization, network_power, power_summary, server_power
from topology import build_fat_tree_like_topology


class PowerModelTests(unittest.TestCase):
    def test_server_power_uses_linear_utilization_formula(self) -> None:
        data_center = build_fat_tree_like_topology(num_pms=2)
        vm = VirtualMachine(id=0, cpu=50.0, ram=32.0)
        self.assertTrue(data_center.pms[0].place(vm))

        expected = P_IDLE + (P_MAX - P_IDLE) * 0.5
        self.assertAlmostEqual(server_power(data_center.pms), expected)

    def test_switch_power_uses_chassis_plus_active_ports(self) -> None:
        data_center = build_fat_tree_like_topology(num_pms=20)
        placement = {0: 0, 1: 1}
        traffic = {(0, 1): 1.0}

        # Same ToR traffic activates one ToR. The ToR has two PM ports plus
        # two traffic path ports counted active by the topology runtime model.
        expected = 45.0 + 4 * 2.0
        self.assertAlmostEqual(network_power(data_center, placement, traffic), expected)

    def test_power_summary_reports_total_and_active_components(self) -> None:
        data_center = build_fat_tree_like_topology(num_pms=20)
        vm_a = VirtualMachine(id=0, cpu=40.0, ram=20.0)
        vm_b = VirtualMachine(id=1, cpu=20.0, ram=20.0)
        self.assertTrue(data_center.pms[0].place(vm_a))
        self.assertTrue(data_center.pms[1].place(vm_b))
        placement = {0: 0, 1: 1}
        traffic = {(0, 1): 1.0}

        summary = power_summary(data_center, data_center.pms, placement, traffic)

        self.assertEqual(summary.active_pms, 2)
        self.assertEqual(summary.active_switches, 1)
        self.assertAlmostEqual(summary.total_dc_power, summary.total_server_power + summary.total_switch_power)
        self.assertAlmostEqual(summary.average_utilization, average_pm_utilization(data_center.pms))


if __name__ == "__main__":
    unittest.main()
