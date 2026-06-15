from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models import AlgorithmResult, VirtualMachine
from simulation import evaluate_result, results_to_dataframe, save_results_csv
from topology import build_fat_tree_like_topology


class EvaluationMetricTests(unittest.TestCase):
    def test_network_localization_score_is_higher_for_shorter_hops(self) -> None:
        data_center = build_fat_tree_like_topology(num_pms=20)
        vm_a = VirtualMachine(id=0, cpu=10.0, ram=10.0)
        vm_b = VirtualMachine(id=1, cpu=10.0, ram=10.0)
        self.assertTrue(data_center.pms[0].place(vm_a))
        self.assertTrue(data_center.pms[1].place(vm_b))

        result = evaluate_result(
            "same-rack",
            data_center,
            data_center.pms,
            {0: 0, 1: 1},
            {(0, 1): 1.0},
            unplaced_count=0,
            baseline_total_power=1000.0,
        )

        self.assertGreaterEqual(result.network_localization_score, 0.0)
        self.assertLessEqual(result.network_localization_score, 100.0)
        self.assertGreater(result.network_localization_score, 50.0)

    def test_dataframe_marks_best_algorithm_and_exports_comparison_csv(self) -> None:
        results = [
            AlgorithmResult("A", {}, 2, 2, 200.0, 40.0, 240.0, 0.0, 4.0, 0, 0.5, 33.3),
            AlgorithmResult("B", {}, 1, 1, 180.0, 20.0, 200.0, 16.7, 2.0, 0, 0.8, 66.7),
        ]
        df = results_to_dataframe(results)
        self.assertEqual(df.loc[df["best_algorithm"] == "YES", "algorithm"].item(), "B")
        self.assertIn("network_localization_score", df.columns)

        output_dir = os.path.join(ROOT, "results", "test_evaluation_metrics")
        os.makedirs(output_dir, exist_ok=True)
        save_results_csv(df, output_dir)
        self.assertTrue(os.path.exists(os.path.join(output_dir, "results.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "algorithm_comparison.csv")))


if __name__ == "__main__":
    unittest.main()
