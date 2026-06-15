# TA-MOACO VM Placement Simulation

Faculty-ready academic demo for **Traffic-Aware Multi-Objective Ant Colony Optimization (TA-MOACO)** applied to virtual machine placement in a data center.

The system compares TA-MOACO against FFD, ACS-VMP, and Traffic-Aware VMP using a realistic fat-tree style topology, server/switch power models, synthetic or real-world VM traces, and presentation-ready visual reports.

## Installation

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

If Streamlit is unavailable in your Python environment, use the built-in local dashboard:

```bash
python local_dashboard.py
```

Then open:

```text
http://localhost:8501
```

## Project Structure

```text
.
├── app.py                         # Optional Streamlit dashboard
├── local_dashboard.py             # Production fallback dashboard, no Streamlit required
├── requirements.txt
├── README.md
├── sample_real_trace.csv
├── sample_missing_ram_trace.csv
├── src
│   ├── main.py                    # CLI entry point
│   ├── models.py                  # VM, PM, switch, data center, result dataclasses
│   ├── topology.py                # Fat-tree topology, paths, hop count, active switches
│   ├── power_model.py             # Server and switch power calculations
│   ├── real_data.py               # Real trace loading, profiling, traffic inference
│   ├── simulation.py              # Workload generation, algorithm orchestration, metrics
│   ├── visualization.py           # High-resolution PNG chart generation
│   └── algorithms
│       ├── ffd.py
│       ├── acs_vmp.py
│       ├── ta_vmp.py
│       └── ta_moaco.py
├── tests
│   ├── test_power_model.py
│   └── test_evaluation_metrics.py
└── results
    ├── algorithm_comparison.csv
    ├── results.csv
    ├── *.png
    └── runs/
```

## CLI Usage

Run the default demo:

```bash
python src/main.py
```

Recommended faculty-demo run:

```bash
python src/main.py --pms 100 --vms 200 --ants 50 --iterations 15 --seed 42
```

TA-MOACO parameters can be tuned:

```bash
python src/main.py --pms 100 --vms 400 --ants 60 --iterations 20 --alpha 1.0 --beta 2.0 --evaporation 0.1
```

CLI output includes a formatted comparison table and saves:

```text
results/results.csv
results/algorithm_comparison.csv
results/energy_comparison.png
results/server_network_power.png
results/active_pm_comparison.png
results/active_switch_comparison.png
results/hop_count_comparison.png
results/traffic_heatmap.png
results/switch_state_map.png
results/convergence_curve.png
results/placement_map.png
results/rack_utilization.png
results/cpu_utilization.png
results/ram_utilization.png
```

## Dashboard Usage

Recommended dashboard:

```bash
python local_dashboard.py
```

Open:

```text
http://localhost:8501
```

The local dashboard includes:

- dataset preview
- preprocessing status pipeline
- progress bar and ETA
- algorithm selector
- parameter explanation
- metric cards
- formatted comparison table
- chart gallery
- VM placement table
- recommended algorithm section
- HTML report export
- CSV export
- synthetic and real trace modes
- optional parameter sensitivity test
- optional SLA stress test

Optional Streamlit dashboard:

```bash
pip install streamlit
streamlit run app.py
```

If Streamlit is not installed, `app.py` prints instructions for using `local_dashboard.py`.

## Vercel Deployment

This repository includes a Vercel-compatible Python serverless entrypoint:

```text
api/index.py
vercel.json
```

The Vercel version is a public synthetic-workload demo. It runs the same simulation engine and generates comparison tables and charts, but it does not rely on local Windows file paths. For full real-CSV local paths and persistent exported run folders, use:

```bash
python local_dashboard.py
```

Deploy from GitHub:

1. Push the repository to GitHub.
2. Import the repository in Vercel.
3. Keep the default project settings.
4. Vercel will use `vercel.json` and deploy `api/index.py`.

Deploy from CLI:

```bash
npx vercel
```

Production deploy:

```bash
npx vercel --prod
```

Vercel entrypoint:

```text
api/index.py exports top-level app
```

Health check:

```text
/api/health
```

## Dataset Support

### Synthetic Data

Synthetic mode generates:

- VM CPU demand
- VM RAM demand
- clustered VM-to-VM traffic
- low, medium, or peak traffic scenarios

Recommended synthetic settings:

```text
PMs: 100-120
VMs: 200-400
Ants: 50-60
Iterations: 10-20
Scenario: medium or peak
```

### Real-World CSV Data

The real-data loader supports:

- `sample_real_trace.csv`
- Bitbrains-style VM traces
- Alibaba Cluster Trace-style CSV files
- generic CSV datasets

Column detection supports common names such as:

```text
vm_id, cpu, cpu_usage, ram, memory, memory_usage, timestamp,
network_in, network_out, app_id, machine_id
```

If columns are missing:

- RAM can be estimated from CPU
- VM IDs can be generated
- app and timestamp fields can be filled with safe defaults
- traffic can be inferred later
- warnings are shown instead of crashing

Large CSV files are handled with:

- chunked pandas loading
- column pruning
- sample sizes of 200, 400, 800, 1000, 2000, and 5000 VMs
- processed dataset caching under `results/cache/`

Real-data outputs:

```text
results/dataset_profile.csv
results/processed_vms.csv
results/processed_dataset.csv
results/traffic_matrix.csv
results/traffic_statistics.csv
results/traffic_heatmap.png
```

Example dashboard path:

```text
C:\Users\Asus\OneDrive\Documents\Green Computing\sample_real_trace.csv
```

## Algorithm Explanation

The project compares four VM placement algorithms.

### First Fit Decreasing

FFD sorts VMs by resource demand and places each VM onto the first feasible physical machine. It is simple and fast, but it does not consider network traffic locality.

### ACS-VMP

ACS-VMP uses ant colony search with pheromone and resource heuristics. It improves server consolidation but is mostly server-centric and does not strongly optimize network traffic.

### Traffic-Aware VMP

Traffic-Aware VMP places highly communicating VMs close together. It usually improves hop count and switch activity, but it may spread VMs across more PMs and increase server power.

### TA-MOACO

TA-MOACO combines server consolidation and traffic-aware placement in one multi-objective ant colony optimization process.

It uses:

- pheromone matrix
- resource heuristic
- traffic-affinity heuristic
- alpha, default `1.0`
- beta, default `2.0`
- evaporation, default `0.1`
- configurable ants and iterations

Each ant builds a complete VM placement while respecting CPU/RAM capacity. Better placements deposit pheromone so later ants learn useful VM-to-PM assignments.

## TA-MOACO Objective

TA-MOACO minimizes a combined score based on:

- total server power
- total network/switch power
- active physical machines
- traffic-weighted average hop count
- SLA violations/unplaced VMs

The key research idea is that localizing high-traffic VMs can reduce active switches and network power, while resource-aware consolidation keeps server power low.

## Infrastructure and Power Model

The simulated data center includes:

- pods
- racks
- physical machines
- ToR/edge switches
- aggregation switches
- core switches

PM-to-PM hop count:

```text
same PM: 0
same ToR/rack: 2
same pod through aggregation: 4
cross-pod through core: 6
```

Server power:

```text
P_server = P_idle + (P_max - P_idle) * utilization
```

Switch power:

```text
P_switch = chassis_power + active_ports * port_power
```

## Result Metrics

The comparison table reports:

- active PMs
- active switches
- server power
- network/switch power
- total DC power
- energy savings percentage
- average hop count
- network localization score
- average utilization
- SLA violations
- recommended/best algorithm marker

The dashboard automatically highlights the best algorithm, preferring SLA-safe results and then lower total power.

## Presentation Charts

High-resolution PNG charts are generated at 300 DPI:

- energy comparison
- server vs network power
- active PM comparison
- active switch comparison
- hop count comparison
- traffic heatmap
- switch activation matrix
- convergence curve
- VM placement map
- topology placement by rack/ToR
- rack utilization
- CPU utilization
- RAM utilization
- run history
- optional parameter sensitivity
- optional SLA stress test

Each dashboard run saves an HTML report under:

```text
results/runs/<run_id>/report.html
```

## Faculty Presentation Guide

Suggested presentation flow:

1. Open the dashboard at `http://localhost:8501`.
2. Show the experiment controls: PMs, VM sample, ants, iterations, alpha, beta, evaporation, seed, and traffic scenario.
3. Show the dataset preview and explain synthetic versus real trace mode.
4. Run all algorithms with 100 PMs, 200 VMs, 50 ants, and 15 iterations.
5. Present the recommended algorithm section.
6. Use metric cards to explain total power, active switches, hop count, utilization, and SLA status.
7. Show the comparison table and point out the highlighted best row.
8. Show the chart gallery:
   - energy comparison proves total power improvement
   - active switch comparison proves network shutdown
   - hop count and traffic heatmap explain traffic locality
   - convergence curve proves TA-MOACO improves over iterations
   - VM placement and rack utilization show where VMs were placed
9. Open the HTML report export as the final submission artifact.
10. Explain the core conclusion: TA-MOACO balances server consolidation and traffic locality, reducing total data center power.

## Troubleshooting

### `localhost:8501` is not loading

Start the fallback dashboard:

```bash
python local_dashboard.py
```

Then open:

```text
http://localhost:8501
```

If the port is busy, stop the old Python process or restart the terminal.

### Streamlit is not installed

Use:

```bash
python local_dashboard.py
```

The fallback dashboard does not require Streamlit.

### `TypeError: Failed to fetch`

This usually means the local dashboard server is not running or crashed during a run. Restart:

```bash
python local_dashboard.py
```

Then reload the browser.

### Real CSV path does not work

Use an absolute Windows path, for example:

```text
C:\Users\Asus\OneDrive\Documents\Green Computing\sample_real_trace.csv
```

Check that the CSV is not open in another application.

### Large datasets are slow

Use 200 or 400 VM samples for live presentation. Use 1000-5000 VM samples for preprocessing demonstration or offline screenshots.

## Final Verification Commands

```bash
python -m unittest discover -s tests
python src/main.py --pms 100 --vms 200 --ants 10 --iterations 5
python local_dashboard.py
```

Then visit:

```text
http://localhost:8501
```

For real-data verification in the dashboard, select:

```text
Data: Real-world trace CSV
Trace type: Generic CSV trace
Trace CSV path: C:\Users\Asus\OneDrive\Documents\Green Computing\sample_real_trace.csv
```
