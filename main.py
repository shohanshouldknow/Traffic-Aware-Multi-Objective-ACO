from __future__ import annotations

import argparse
import os
import sys

from simulation import generate_traffic_matrix, run_simulation, save_results_csv
from visualization import generate_all_charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TA-MOACO data center VM placement simulation")
    parser.add_argument("--pms", type=int, default=100, help="Number of physical machines")
    parser.add_argument("--vms", type=int, default=200, help="Number of virtual machines, recommended 200-800")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ants", type=int, default=50, help="Number of ants for ACS and TA-MOACO")
    parser.add_argument("--iterations", type=int, default=15, help="TA-MOACO iterations")
    parser.add_argument("--alpha", type=float, default=1.0, help="TA-MOACO pheromone influence")
    parser.add_argument("--beta", type=float, default=2.0, help="TA-MOACO heuristic influence")
    parser.add_argument("--evaporation", type=float, default=0.1, help="TA-MOACO pheromone evaporation rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vms < 200 or args.vms > 800:
        print("Warning: paper demo expects 200-800 VMs; continuing with requested value.")

    print(f"Running simulation with {args.pms} PMs, {args.vms} VMs, {args.ants} ants...")
    df, results, _data_center = run_simulation(
        num_pms=args.pms,
        num_vms=args.vms,
        seed=args.seed,
        ants=args.ants,
        iterations=args.iterations,
        alpha=args.alpha,
        beta=args.beta,
        evaporation=args.evaporation,
    )

    output_dir = "results"
    csv_path = save_results_csv(df, output_dir)
    traffic = generate_traffic_matrix(args.vms, seed=args.seed)
    chart_paths = generate_all_charts(df, results, _data_center, output_dir, traffic=traffic)

    print("\nComparison table:")
    display_df = df.copy()
    if "best_algorithm" in display_df.columns:
        display_df["best_algorithm"] = display_df["best_algorithm"].replace({"YES": "<-- BEST"})
    print(display_df.to_string(index=False))
    print(f"\nSaved CSV: {os.path.abspath(csv_path)}")
    print(f"Saved comparison CSV: {os.path.abspath(os.path.join(output_dir, 'algorithm_comparison.csv'))}")
    for path in chart_paths:
        print(f"Saved chart: {os.path.abspath(path)}")

    best = min(results, key=lambda result: result.total_power)
    print(f"\nRecommended placement: {best.name} with total power {best.total_power:.2f} W")
    print("TA-MOACO localizes high-traffic VMs, reduces traffic-weighted hops, and allows unused switches to stay off.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Simulation interrupted.")
