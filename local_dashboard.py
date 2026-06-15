from __future__ import annotations

import json
import os
import sys
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
RESULTS_ROOT = os.path.abspath(os.environ.get("TA_MOACO_RESULTS_DIR", os.path.join(ROOT, "results")))

from simulation import run_simulation, save_results_csv
from simulation import generate_traffic_matrix, generate_vms
from algorithms.ffd import run_ffd
from algorithms.ta_moaco import run_ta_moaco
from models import VirtualMachine
from power_model import network_power, server_power
from real_data import load_alibaba_trace, load_bitbrains_trace, load_custom_real_trace, save_processed_outputs, save_traffic_outputs
from topology import build_fat_tree_like_topology
from visualization import generate_all_charts


PORT = 8501


def estimate_seconds(pms: int, vms: int, ants: int, iterations: int) -> int:
    """Small runtime estimator for the browser progress display."""

    aco_work = max(ants, 1) * (max(8, iterations // 2) + max(iterations, 1))
    baseline_work = 50 * (8 + 15)
    scale = (max(vms, 1) / 200.0) * (aco_work / baseline_work)
    return max(35, int(110 * scale))


def html_page(body: str) -> bytes:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TA-MOACO Dashboard</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #637083;
      --line: #d9e2ec;
      --line-strong: #b7c7d7;
      --paper: rgba(255,255,255,.92);
      --bg: #f5f8fc;
      --navy: #0f2742;
      --blue: #246bfe;
      --cyan: #00a6c8;
      --green: #16845f;
      --amber: #c67a12;
      --red: #d84a4a;
      --violet: #5956d6;
      --shadow: 0 22px 70px rgba(35, 61, 93, .13);
      --soft-shadow: 0 10px 28px rgba(35, 61, 93, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", "Aptos", "Trebuchet MS", sans-serif;
      margin: 0;
      background:
        linear-gradient(135deg, rgba(36,107,254,.07), transparent 34%),
        linear-gradient(0deg, rgba(255,255,255,.78), rgba(255,255,255,.78)),
        repeating-linear-gradient(90deg, rgba(15,39,66,.045) 0 1px, transparent 1px 84px),
        repeating-linear-gradient(0deg, rgba(15,39,66,.035) 0 1px, transparent 1px 84px);
      color: var(--ink);
    }}
    header {{
      position: relative;
      overflow: hidden;
      background:
        linear-gradient(135deg, rgba(15,39,66,.98), rgba(27,92,132,.92)),
        repeating-linear-gradient(90deg, rgba(255,255,255,.08) 0 1px, transparent 1px 62px);
      color: white;
      padding: 34px 46px 36px;
      border-bottom: 1px solid rgba(255,255,255,.16);
    }}
    .hero {{ position: relative; max-width: 1240px; margin: 0 auto; z-index: 1; }}
    .eyebrow {{ margin: 0 0 14px; color: #bdeafe; font-size: 13px; letter-spacing: .12em; text-transform: uppercase; font-weight: 800; }}
    header h1 {{ margin: 0 0 12px; max-width: 820px; font-size: 48px; line-height: 1.04; }}
    header p {{ margin: 0; max-width: 760px; font-size: 20px; color: #e8f4ff; }}
    .hero-badges {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }}
    .hero-badges span {{ border: 1px solid rgba(255,255,255,.28); background: rgba(255,255,255,.12); border-radius: 999px; padding: 8px 12px; font-weight: 700; font-size: 13px; }}
    main {{ padding: 30px 42px; max-width: 1280px; margin: auto; }}
    form, section {{
      background: var(--paper);
      border: 1px solid rgba(183,199,215,.78);
      border-radius: 14px;
      padding: 26px;
      margin-bottom: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }}
    h2 {{ margin: 0 0 18px; font-size: 28px; letter-spacing: -.02em; }}
    label {{ display: inline-flex; align-items: center; gap: 9px; margin: 8px 18px 8px 0; font-size: 16px; font-weight: 750; color: #2c3748; }}
    input, select {{ padding: 11px 12px; width: 112px; font-size: 16px; border: 1px solid var(--line-strong); border-radius: 8px; background: #fbfdff; color: var(--ink); }}
    select {{ width: 150px; }}
    input.path-input {{ width: min(520px, 92vw); }}
    input.check {{ width: 18px; height: 18px; accent-color: var(--blue); }}
    input:focus, select:focus {{ outline: 3px solid rgba(36,107,254,.18); border-color: var(--blue); }}
    button {{ background: linear-gradient(135deg, var(--blue), var(--cyan)); color: white; border: 0; border-radius: 9px; padding: 13px 19px; cursor: pointer; font-size: 16px; font-weight: 800; box-shadow: 0 14px 28px rgba(36,107,254,.23); }}
    button:hover {{ transform: translateY(-1px); box-shadow: 0 18px 34px rgba(36,107,254,.29); }}
    button:disabled {{ opacity: 0.65; cursor: wait; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 14px; overflow: hidden; border-radius: 10px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 10px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #edf4fb; color: #223047; }}
    tr:nth-child(even) td {{ background: #fbfdff; }}
    tr.best-row td {{ background: #e8fff5 !important; border-top: 2px solid #34d399; border-bottom: 2px solid #34d399; font-weight: 800; }}
    img {{ width: 100%; background: white; border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--soft-shadow); }}
    code {{ background: #eef4fb; padding: 2px 6px; border-radius: 6px; }}
    .hint {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 18px; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    .figure {{ background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: 10px; box-shadow: var(--soft-shadow); }}
    .figure img {{ border: 0; box-shadow: none; border-radius: 8px; }}
    .figure figcaption {{ color: var(--muted); font-size: 13px; margin: 8px 4px 2px; font-weight: 700; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid var(--line); border-radius: 12px; padding: 18px; background: linear-gradient(180deg, #ffffff, #f8fbff); box-shadow: var(--soft-shadow); }}
    .card h3 {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin: 0 0 12px; font-size: 20px; }}
    .metric {{ display: flex; justify-content: space-between; gap: 12px; margin: 9px 0; }}
    .metric span:first-child {{ color: var(--muted); }}
    .bar {{ height: 11px; border-radius: 999px; background: #e5edf6; overflow: hidden; margin: 9px 0 15px; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--green), var(--blue)); }}
    .progress-wrap {{ display: none; }}
    .progress-track {{ height: 18px; background: #e5edf6; border-radius: 999px; overflow: hidden; border: 1px solid #d1e1f1; }}
    .progress-fill {{ width: 0%; height: 100%; background: linear-gradient(90deg, var(--blue), var(--cyan), var(--green)); transition: width .4s ease; }}
    .status-row {{ display: flex; flex-wrap: wrap; gap: 18px; margin-top: 12px; color: var(--muted); }}
    .placement-table {{ max-height: 360px; overflow: auto; }}
    .pill {{ display: inline-block; background: #e8fff5; color: #0f684b; border: 1px solid #a7f3d0; padding: 6px 10px; border-radius: 999px; font-weight: 800; font-size: 13px; }}
    .pill-blue {{ background: #ecf4ff; color: #1751bf; border-color: #bdd5ff; }}
    .pill-amber {{ background: #fff7e8; color: #8a5208; border-color: #f5cf8d; }}
    .panel-title {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
    .feature {{ border: 1px solid var(--line); border-radius: 12px; padding: 16px; background: #fbfdff; }}
    .feature strong {{ display: block; margin-bottom: 8px; color: var(--navy); }}
    .run-layout {{ display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: end; }}
    .inputs {{ display: flex; flex-wrap: wrap; gap: 2px 0; }}
    .input-group {{ margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--line); }}
    .input-group h3 {{ margin: 0 0 8px; font-size: 15px; color: var(--navy); text-transform: uppercase; letter-spacing: .08em; }}
    .algorithm-picker {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .algorithm-picker label {{ margin: 0; border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; background: #fff; box-shadow: 0 6px 18px rgba(35,61,93,.06); }}
    .table-wrap {{ width: 100%; overflow: auto; border: 1px solid var(--line); border-radius: 12px; background: #fff; }}
    .preview-table {{ max-height: 340px; overflow: auto; }}
    .run-summary {{ display: grid; grid-template-columns: minmax(260px, 1.1fr) repeat(3, minmax(150px, .6fr)); gap: 14px; align-items: stretch; }}
    .hero-metric {{ border: 1px solid var(--line); border-radius: 12px; padding: 16px; background: linear-gradient(180deg, #ffffff, #f5fbff); }}
    .hero-metric span {{ display: block; color: var(--muted); font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; }}
    .hero-metric strong {{ display: block; margin-top: 8px; font-size: 26px; color: var(--navy); }}
    .pipeline {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin-top: 16px; }}
    .step {{ border: 1px solid var(--line); border-radius: 12px; padding: 14px; background: #fff; }}
    .step b {{ display: block; color: var(--navy); margin-bottom: 5px; }}
    .step span {{ color: var(--muted); font-size: 13px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    a.button-link {{ display: inline-block; text-decoration: none; background: var(--navy); color: white; border-radius: 9px; padding: 11px 14px; font-weight: 800; }}
    details.explain {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; background: #fbfdff; }}
    details.explain summary {{ cursor: pointer; font-weight: 800; color: var(--navy); }}
    @media (max-width: 720px) {{
      header {{ padding: 28px 22px; }}
      header h1 {{ font-size: 36px; }}
      main {{ padding: 22px; }}
      .grid, .gallery, .run-summary {{ grid-template-columns: 1fr; }}
      .run-layout {{ grid-template-columns: 1fr; }}
      label {{ width: 100%; justify-content: space-between; }}
      input, select {{ width: 52%; }}
      .algorithm-picker label {{ width: auto; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="hero">
      <p class="eyebrow">Green Computing Research Simulator</p>
      <h1>TA-MOACO Intelligent VM Placement Console</h1>
      <p>Analyze server power, network power, hop count, SLA status, and VM-to-PM placement in one faculty-ready visual dashboard.</p>
      <div class="hero-badges">
        <span>Fat-tree topology</span>
        <span>50-ant ACO ready</span>
        <span>Traffic-aware placement</span>
        <span>Live ETA</span>
        <span>Presentation graphs</span>
      </div>
    </div>
  </header>
  <main>{body}</main>
  <script>
    const form = document.getElementById("run-form");
    const progressBox = document.getElementById("progress-box");
    const progressFill = document.getElementById("progress-fill");
    const elapsedNode = document.getElementById("elapsed");
    const etaNode = document.getElementById("eta");
    const percentNode = document.getElementById("percent");
    const statusNode = document.getElementById("status-text");
    const resultsNode = document.getElementById("results");

    function fmt(seconds) {{
      seconds = Math.max(0, Math.round(seconds));
      const m = Math.floor(seconds / 60);
      const s = seconds % 60;
      return m + ":" + String(s).padStart(2, "0");
    }}

    if (form) {{
      form.addEventListener("submit", async (event) => {{
        event.preventDefault();
        const button = form.querySelector("button");
        const data = new FormData(form);
        const params = new URLSearchParams(data);
        const pms = Number(data.get("pms"));
        const vms = Number(data.get("vms"));
        const ants = Number(data.get("ants"));
        const iterations = Number(data.get("iterations"));
        const sensitivity = data.get("sensitivity") === "on";
        const stress = data.get("stress") === "on";
        const realData = data.get("data_source") === "real";
        const acoWork = ants * (Math.max(8, Math.floor(iterations / 2)) + iterations);
        let estimated = Math.max(35, Math.round(110 * (vms / 200) * (acoWork / (50 * (8 + 15)))));
        if (sensitivity) estimated += 20;
        if (stress) estimated += 20;
        if (realData) estimated += 10;
        const start = Date.now();
        progressBox.style.display = "block";
        resultsNode.innerHTML = "";
        button.disabled = true;
        statusNode.textContent = realData
          ? "Preprocessing dataset: loading chunks, checking cache, and normalizing VMs..."
          : "Preparing synthetic workload...";
        const statusPlan = [
          [1800, realData ? "Inferring VM traffic from application, timestamp, CPU, RAM, and network columns..." : "Generating clustered VM traffic matrix..."],
          [4200, "Running selected placement algorithms with CPU/RAM constraints..."],
          [9000, "Evaluating power, hops, SLA violations, utilization, and localization score..."],
          [13000, "Rendering comparison charts, placement maps, and HTML export..."]
        ];
        statusPlan.forEach(([delay, message]) => {{
          setTimeout(() => {{
            if (button.disabled) statusNode.textContent = message;
          }}, delay);
        }});

        const timer = setInterval(() => {{
          const elapsed = (Date.now() - start) / 1000;
          const pct = Math.min(95, Math.round((elapsed / estimated) * 100));
          progressFill.style.width = pct + "%";
          percentNode.textContent = pct + "%";
          elapsedNode.textContent = fmt(elapsed);
          etaNode.textContent = fmt(Math.max(0, estimated - elapsed));
        }}, 500);

        try {{
          const response = await fetch("/api/run?" + params.toString());
          const payload = await response.json();
          clearInterval(timer);
          progressFill.style.width = "100%";
          percentNode.textContent = "100%";
          elapsedNode.textContent = fmt((Date.now() - start) / 1000);
          etaNode.textContent = "0:00";
          statusNode.textContent = "Completed.";
          if (!payload.ok) {{
            resultsNode.innerHTML = "<section><h2>Error</h2><pre>" + payload.error + "</pre></section>";
          }} else {{
            resultsNode.innerHTML = payload.html;
          }}
        }} catch (error) {{
          clearInterval(timer);
          statusNode.textContent = "Failed.";
          resultsNode.innerHTML = "<section><h2>Error</h2><pre>" + error + "</pre></section>";
        }} finally {{
          button.disabled = false;
        }}
      }});
    }}
  </script>
</body>
</html>""".encode("utf-8")


def table_html(df) -> str:
    """Render the algorithm table and highlight the selected best row."""

    display_df = df.copy()
    if "best_algorithm" in display_df.columns:
        display_df["best_algorithm"] = display_df["best_algorithm"].replace({"YES": "Best"})
    html = display_df.to_html(index=False, border=0, classes="results", escape=False)
    rows = html.split("<tr>")
    highlighted = [rows[0]]
    for row in rows[1:]:
        if "<td>Best</td>" in row:
            highlighted.append('<tr class="best-row">' + row)
        else:
            highlighted.append("<tr>" + row)
    return "".join(highlighted)


def placement_table_html(best) -> str:
    rows = []
    for vm_id, pm_id in sorted(best.placement.items())[:350]:
        rows.append(f"<tr><td>{vm_id}</td><td>{pm_id}</td></tr>")
    return f"""
<div class="placement-table">
  <table>
    <thead><tr><th>VM ID</th><th>Assigned PM ID</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
<p class="hint">Showing first 350 VM assignments. Full placement contains {len(best.placement)} VMs.</p>
"""


def checked(value: bool) -> str:
    """Return an HTML checked attribute when a checkbox option is enabled."""

    return "checked" if value else ""


ALGORITHMS = {
    "FFD": "alg_ffd",
    "ACS-VMP": "alg_acs",
    "Traffic-Aware VMP": "alg_ta",
    "TA-MOACO": "alg_moaco",
}


def selected_algorithm_names(query: dict[str, list[str]]) -> list[str]:
    """Return selected algorithms while keeping all algorithms enabled by default."""

    if not any(field in query for field in ALGORITHMS.values()):
        return list(ALGORITHMS.keys())
    selected = [name for name, field in ALGORITHMS.items() if query.get(field, ["off"])[0] == "on"]
    return selected or list(ALGORITHMS.keys())


def algorithm_selector_html(query: dict[str, list[str]]) -> str:
    """Render checkbox controls for selecting algorithms shown in results."""

    selected = set(selected_algorithm_names(query))
    items = []
    for name, field in ALGORITHMS.items():
        items.append(
            f'<label><input class="check" name="{field}" type="checkbox" {checked(name in selected)}> {name}</label>'
        )
    return f'<div class="algorithm-picker">{"".join(items)}</div>'


def dataset_preview_html(query: dict[str, list[str]]) -> str:
    """Show a lightweight preview of synthetic or real trace input before running."""

    vms = int(query.get("vms", ["200"])[0])
    seed = int(query.get("seed", ["42"])[0])
    scenario = query.get("scenario", ["medium"])[0]
    data_source = query.get("data_source", ["synthetic"])[0]
    trace_path = query.get("trace_path", [""])[0].strip().strip('"')

    if data_source == "real" and trace_path:
        if not os.path.exists(trace_path):
            return f"""
<section>
  <div class="panel-title">
    <div><h2>Dataset Preview</h2><p class="hint">Real trace mode is selected, but the CSV path was not found.</p></div>
    <span class="pill pill-amber">Waiting for file</span>
  </div>
  <p><code>{escape(trace_path)}</code></p>
</section>
"""
        try:
            preview = pd.read_csv(trace_path, nrows=8)
            columns = ", ".join(escape(str(col)) for col in preview.columns[:14])
            return f"""
<section>
  <div class="panel-title">
    <div><h2>Dataset Preview</h2><p class="hint">First rows from the selected real-world CSV. Full preprocessing happens when you run the simulation.</p></div>
    <span class="pill pill-blue">Real trace</span>
  </div>
  <div class="mini-grid">
    <div class="feature"><strong>Path</strong><code>{escape(trace_path)}</code></div>
    <div class="feature"><strong>Detected columns</strong>{columns}</div>
    <div class="feature"><strong>Sample target</strong>{vms} VMs after chunked preprocessing.</div>
  </div>
  <div class="table-wrap preview-table">{preview.to_html(index=False, border=0, classes="results")}</div>
</section>
"""
        except Exception as exc:
            return f"""
<section>
  <div class="panel-title">
    <div><h2>Dataset Preview</h2><p class="hint">Could not preview the CSV. The run step will report the exact loader error if preprocessing fails.</p></div>
    <span class="pill pill-amber">Preview warning</span>
  </div>
  <pre>{escape(repr(exc))}</pre>
</section>
"""

    preview_vms = generate_vms(min(vms, 8), seed=seed)
    preview_traffic_vms = min(vms, 200)
    traffic = generate_traffic_matrix(preview_traffic_vms, seed=seed, profile=scenario)
    preview_df = pd.DataFrame(
        [{"vm_id": vm.id, "cpu": vm.cpu, "ram": vm.ram} for vm in preview_vms]
    )
    traffic_values = list(traffic.values())
    mean_traffic = sum(traffic_values) / len(traffic_values) if traffic_values else 0.0
    return f"""
<section>
  <div class="panel-title">
    <div><h2>Dataset Preview</h2><p class="hint">Synthetic preview generated from the current VM sample, seed, and traffic scenario.</p></div>
    <span class="pill pill-blue">Synthetic</span>
  </div>
  <div class="mini-grid">
    <div class="feature"><strong>VM sample size</strong>{vms} requested VMs.</div>
    <div class="feature"><strong>Traffic scenario</strong>{escape(scenario)} profile preview with {len(traffic)} links across {preview_traffic_vms} sampled VMs.</div>
    <div class="feature"><strong>Mean traffic</strong>{mean_traffic:.3f} average demand across generated links.</div>
    <div class="feature"><strong>Seed</strong>{seed} for reproducible preview and run.</div>
  </div>
  <div class="table-wrap preview-table">{preview_df.to_html(index=False, border=0, classes="results")}</div>
</section>
"""


def figure_html(image_base: str, filename: str, title: str, caption: str, output_dir: str) -> str:
    """Render one chart card only when the corresponding PNG exists."""

    if not os.path.exists(os.path.join(output_dir, filename)):
        return ""
    return f"""
<figure class="figure">
  <img src="{image_base}/{filename}" alt="{escape(title)}">
  <figcaption>{escape(title)} - {escape(caption)}</figcaption>
</figure>
"""


def chart_gallery_html(image_base: str, output_dir: str, trace_bundle=None, include_sensitivity: bool = False, include_stress: bool = False, time_window_df=None) -> str:
    """Build the responsive chart gallery from generated PNG files."""

    core_items = [
        ("energy_comparison.png", "Energy comparison", "stacked server and network power"),
        ("server_network_power.png", "Server vs network power", "side-by-side power breakdown"),
        ("active_pm_comparison.png", "Active PM comparison", "server consolidation by algorithm"),
        ("active_switch_comparison.png", "Active switch comparison", "network shutdown opportunities"),
        ("hop_count_comparison.png", "Hop count comparison", "traffic-weighted network distance"),
        ("placement_map.png", "VM placement map", "VM density across physical machines"),
        ("convergence_curve.png", "Convergence curve", "TA-MOACO objective improvement"),
        ("traffic_heatmap.png", "Traffic heatmap", "VM-to-VM communication intensity"),
        ("switch_state_map.png", "Switch activation matrix", "active and powered-down switches"),
        ("rack_utilization.png", "Rack utilization", "average utilization per rack"),
        ("cpu_utilization.png", "CPU utilization", "per-PM CPU utilization"),
        ("ram_utilization.png", "RAM utilization", "per-PM RAM utilization"),
        ("topology_placement.png", "Topology placement", "VM placement grouped by ToR"),
        ("run_history.png", "Run history", "recent TA-MOACO versus FFD runs"),
    ]
    if trace_bundle:
        core_items.append(("dataset_profile.png", "Dataset profile", "real trace CPU/RAM distribution"))
    if time_window_df is not None and not time_window_df.empty:
        core_items.append(("time_window_energy.png", "Time-window energy", "real trace windows over time"))
    if include_sensitivity:
        core_items.append(("parameter_sensitivity.png", "Parameter sensitivity", "TA-MOACO tuning comparison"))
    if include_stress:
        core_items.append(("sla_stress_test.png", "SLA stress test", "violations under increased load"))

    cards = [figure_html(image_base, filename, title, caption, output_dir) for filename, title, caption in core_items]
    return f'<div class="gallery">{"".join(cards)}</div>'


def append_run_history(output_dir: str, df, query: dict[str, list[str]], elapsed: float):
    """Append a compact row to the global and per-run history CSV files."""

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    history_path = os.path.join(RESULTS_ROOT, "run_history.csv")
    ffd = df[df["algorithm"] == "FFD"].iloc[0]
    moaco = df[df["algorithm"] == "TA-MOACO"].iloc[0]
    row = {
        "timestamp": int(time.time()),
        "pms": int(query.get("pms", ["100"])[0]),
        "vms": int(query.get("vms", ["200"])[0]),
        "ants": int(query.get("ants", ["50"])[0]),
        "iterations": int(query.get("iterations", ["15"])[0]),
        "scenario": query.get("scenario", ["medium"])[0],
        "elapsed_seconds": round(elapsed, 2),
        "ffd_total_power_w": float(ffd["total_power_w"]),
        "ta_moaco_total_power_w": float(moaco["total_power_w"]),
        "ta_moaco_savings_pct": float(moaco["energy_savings_pct"]),
    }
    old = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()
    history = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(history_path, index=False)
    history.to_csv(os.path.join(output_dir, "run_history.csv"), index=False)
    return history


def run_parameter_sensitivity(vms, traffic, pms: int, seed: int):
    """Run a lightweight TA-MOACO parameter sweep for dashboard diagnostics."""

    variants = [
        {"variant": "alpha 0.7", "alpha": 0.7, "beta": 2.0, "evaporation": 0.1},
        {"variant": "alpha 1.0", "alpha": 1.0, "beta": 2.0, "evaporation": 0.1},
        {"variant": "beta 1.5", "alpha": 1.0, "beta": 1.5, "evaporation": 0.1},
        {"variant": "beta 2.5", "alpha": 1.0, "beta": 2.5, "evaporation": 0.1},
        {"variant": "evap 0.2", "alpha": 1.0, "beta": 2.0, "evaporation": 0.2},
    ]
    rows = []
    for index, variant in enumerate(variants):
        data_center = build_fat_tree_like_topology(num_pms=pms)
        placement, placed_pms, unplaced = run_ta_moaco(
            data_center.pms,
            vms,
            traffic,
            data_center,
            ants=12,
            iterations=5,
            alpha=variant["alpha"],
            beta=variant["beta"],
            evaporation=variant["evaporation"],
            seed=seed + 100 + index,
        )
        rows.append(
            {
                "variant": variant["variant"],
                "alpha": variant["alpha"],
                "beta": variant["beta"],
                "evaporation": variant["evaporation"],
                "total_power_w": round(server_power(placed_pms) + network_power(data_center, placement, traffic), 2),
                "unplaced_vms": unplaced,
            }
        )
    return pd.DataFrame(rows)


def run_sla_stress(vms, traffic, pms: int, seed: int):
    """Increase VM demand and compare SLA violations for FFD and TA-MOACO."""

    rows = []
    for factor in [1.0, 1.15, 1.3, 1.45, 1.6]:
        stressed_vms = [
            VirtualMachine(vm.id, min(95.0, vm.cpu * factor), min(245.0, vm.ram * factor))
            for vm in vms
        ]
        data_center = build_fat_tree_like_topology(num_pms=pms)
        ffd_placement, ffd_pms, ffd_unplaced = run_ffd(data_center.pms, stressed_vms)
        rows.append({"algorithm": "FFD", "load_multiplier": factor, "sla_violations": ffd_unplaced})

        data_center = build_fat_tree_like_topology(num_pms=pms)
        moaco_placement, moaco_pms, moaco_unplaced = run_ta_moaco(
            data_center.pms,
            stressed_vms,
            traffic,
            data_center,
            ants=12,
            iterations=5,
            seed=seed + 200,
        )
        rows.append({"algorithm": "TA-MOACO", "load_multiplier": factor, "sla_violations": moaco_unplaced})
    return pd.DataFrame(rows)


def run_time_window_simulation(trace_bundle, pms: int, scenario: str, seed: int):
    """Run a compact energy comparison across real trace timestamps."""

    if trace_bundle is None or trace_bundle.raw_frame["timestamp"].nunique() <= 1:
        return pd.DataFrame()

    rows = []
    windows = sorted(trace_bundle.raw_frame["timestamp"].unique())[:6]
    for offset, window in enumerate(windows):
        frame = trace_bundle.raw_frame[trace_bundle.raw_frame["timestamp"] == window].copy().head(220)
        if len(frame) < 2:
            continue
        frame = frame.reset_index(drop=True)
        frame["sim_vm_id"] = range(len(frame))
        vms = [
            VirtualMachine(id=int(row.sim_vm_id), cpu=float(row.cpu), ram=float(row.ram))
            for row in frame.itertuples()
        ]
        from real_data import infer_traffic_from_apps

        traffic = infer_traffic_from_apps(frame, profile=scenario)
        data_center = build_fat_tree_like_topology(num_pms=pms)
        ffd_placement, ffd_pms, _ffd_unplaced = run_ffd(data_center.pms, vms)
        rows.append(
            {
                "window": str(window),
                "algorithm": "FFD",
                "vm_count": len(vms),
                "total_power_w": round(server_power(ffd_pms) + network_power(data_center, ffd_placement, traffic), 2),
            }
        )

        data_center = build_fat_tree_like_topology(num_pms=pms)
        moaco_placement, moaco_pms, _moaco_unplaced = run_ta_moaco(
            data_center.pms,
            vms,
            traffic,
            data_center,
            ants=10,
            iterations=5,
            seed=seed + 300 + offset,
        )
        rows.append(
            {
                "window": str(window),
                "algorithm": "TA-MOACO",
                "vm_count": len(vms),
                "total_power_w": round(server_power(moaco_pms) + network_power(data_center, moaco_placement, traffic), 2),
            }
        )
    return pd.DataFrame(rows)


def create_report(output_dir: str, df, best, run_id: str, query: dict[str, list[str]], trace_bundle=None, time_window_df=None) -> str:
    """Write a self-contained HTML report for a completed dashboard run."""

    report_path = os.path.join(output_dir, "report.html")
    scenario = query.get("scenario", ["medium"])[0]
    real_section = ""
    if trace_bundle:
        profile = trace_bundle.profile
        real_section = f"""
<h2>Real-World Trace Processing</h2>
<p><strong>Source:</strong> {profile['source_name']}</p>
<p><strong>Traffic method:</strong> {profile['traffic_mode']}</p>
<p><strong>VMs:</strong> {profile['vm_count']} | <strong>Applications:</strong> {profile['app_count']} | <strong>Time windows:</strong> {profile['time_windows']}</p>
<p><strong>CPU average/peak:</strong> {profile['avg_cpu']} / {profile['peak_cpu']} | <strong>RAM average/peak:</strong> {profile['avg_ram']} / {profile['peak_ram']}</p>
<p><strong>Traffic inference:</strong> normalized 0-1 matrix from app, timestamp, CPU, RAM, network usage, correlation score, and small deterministic noise.</p>
<p><strong>Traffic min/mean/max:</strong> {profile.get('traffic_min', 0)} / {profile.get('traffic_mean', 0)} / {profile.get('traffic_max', 0)}</p>
<h2>Real Dataset Profile</h2><img src="dataset_profile.png">
"""
        if time_window_df is not None and not time_window_df.empty:
            real_section += "<h2>Time-Window Energy Simulation</h2><img src=\"time_window_energy.png\">"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>TA-MOACO Report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:36px;color:#172033}}img{{max-width:100%;border:1px solid #d9e2ec;border-radius:10px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d1d5db;padding:8px}}</style>
</head><body>
<h1>TA-MOACO Simulation Report</h1>
<p><strong>Run:</strong> {run_id}</p>
<p><strong>Scenario:</strong> {scenario}</p>
<p><strong>Recommended:</strong> {best.name}, total power {best.total_power:.2f} W, savings {best.energy_savings_pct:.2f}%.</p>
{df.to_html(index=False)}
{real_section}
<h2>Energy Comparison</h2><img src="energy_comparison.png">
<h2>Server vs Network Power</h2><img src="server_network_power.png">
<h2>Active PM Comparison</h2><img src="active_pm_comparison.png">
<h2>Active Switch Comparison</h2><img src="active_switch_comparison.png">
<h2>TA-MOACO Convergence</h2><img src="convergence_curve.png">
<h2>Traffic Heatmap</h2><img src="traffic_heatmap.png">
<h2>Active Switch Map</h2><img src="switch_state_map.png">
<h2>VM Placement by ToR</h2><img src="topology_placement.png">
<h2>Rack Utilization</h2><img src="rack_utilization.png">
<h2>CPU Utilization</h2><img src="cpu_utilization.png">
<h2>RAM Utilization</h2><img src="ram_utilization.png">
</body></html>"""
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(html)
    return report_path


def result_cards_html(df) -> str:
    """Render per-algorithm metric cards for the dashboard results area."""

    min_total = max(1.0, float(df["total_power_w"].min()))
    max_total = max(min_total, float(df["total_power_w"].max()))
    cards = []
    for row in df.to_dict("records"):
        width = 100.0 - ((float(row["total_power_w"]) - min_total) / max(1.0, max_total - min_total) * 55.0)
        is_best = row.get("best_algorithm") == "YES" or float(row["total_power_w"]) == min_total
        badge = '<span class="pill">Recommended</span>' if is_best else ""
        cards.append(
            f"""
<div class="card">
  <h3>{row['algorithm']} {badge}</h3>
  <div class="metric"><span>Total power</span><strong>{row['total_power_w']} W</strong></div>
  <div class="bar"><span style="width:{width:.1f}%"></span></div>
  <div class="metric"><span>Total server power</span><strong>{row['server_power_w']} W</strong></div>
  <div class="metric"><span>Total switch power</span><strong>{row.get('switch_power_w', row['network_power_w'])} W</strong></div>
  <div class="metric"><span>Total DC power</span><strong>{row.get('total_dc_power_w', row['total_power_w'])} W</strong></div>
  <div class="metric"><span>Active PMs</span><strong>{row['active_pms']}</strong></div>
  <div class="metric"><span>Active switches</span><strong>{row['active_switches']}</strong></div>
  <div class="metric"><span>Average utilization</span><strong>{row.get('average_utilization_pct', 0)}%</strong></div>
  <div class="metric"><span>Localization score</span><strong>{row.get('network_localization_score', 0)}/100</strong></div>
  <div class="metric"><span>Average hops</span><strong>{row['average_hop_count']}</strong></div>
  <div class="metric"><span>SLA violations</span><strong>{row['sla_violations']}</strong></div>
  <div class="metric"><span>Energy savings</span><strong>{row['energy_savings_pct']}%</strong></div>
</div>
"""
        )
    return f'<div class="cards">{"".join(cards)}</div>'


def real_trace_summary_html(trace_bundle) -> str:
    """Render a summary of normalized real-world trace characteristics."""

    profile = trace_bundle.profile
    warning_html = "".join(f"<li>{warning}</li>" for warning in trace_bundle.warnings)
    warning_section = f"<ul>{warning_html}</ul>" if warning_html else "<p class=\"hint\">No dataset warnings.</p>"
    return f"""
<section>
  <div class="panel-title">
    <div>
      <h2>Real-World Trace Profile</h2>
      <p class="hint">The uploaded trace was normalized into VM CPU/RAM demand and an inferred traffic matrix.</p>
    </div>
    <span class="pill">{profile['source_name']}</span>
  </div>
  <div class="mini-grid">
    <div class="feature"><strong>VM count</strong>{profile['vm_count']} normalized VMs used for placement.</div>
    <div class="feature"><strong>Application groups</strong>{profile['app_count']} app/job groups used for traffic inference.</div>
    <div class="feature"><strong>Time windows</strong>{profile['time_windows']} unique timestamps found in the trace.</div>
    <div class="feature"><strong>CPU demand</strong>Average {profile['avg_cpu']}, peak {profile['peak_cpu']}.</div>
    <div class="feature"><strong>RAM demand</strong>Average {profile['avg_ram']}, peak {profile['peak_ram']}.</div>
    <div class="feature"><strong>Traffic mode</strong>{profile['traffic_mode']} with {profile['traffic_edges']} inferred VM links.</div>
    <div class="feature"><strong>Traffic range</strong>Min {profile.get('traffic_min', 0)}, mean {profile.get('traffic_mean', 0)}, max {profile.get('traffic_max', 0)} after 0-1 normalization.</div>
    <div class="feature"><strong>Cache</strong>{profile.get('cache_status', 'unknown')} cache status for this processed dataset.</div>
    <div class="feature"><strong>Rows loaded</strong>{profile.get('rows_loaded', profile['vm_count'])} rows loaded using {profile.get('chunks_read', 'n/a')} chunk(s).</div>
    <div class="feature"><strong>Memory used</strong>{profile.get('memory_mb', 'n/a')} MB for the sampled preprocessing frame.</div>
  </div>
  <h3>Loader Warnings</h3>
  {warning_section}
</section>
"""


def run_simulation_html(query: dict[str, list[str]]) -> str:
    """Execute a dashboard run and return the results section as HTML."""

    pms = int(query.get("pms", ["100"])[0])
    vms = int(query.get("vms", ["200"])[0])
    ants = int(query.get("ants", ["50"])[0])
    iterations = int(query.get("iterations", ["15"])[0])
    alpha = float(query.get("alpha", ["1.0"])[0])
    beta = float(query.get("beta", ["2.0"])[0])
    evaporation = float(query.get("evaporation", ["0.1"])[0])
    seed = int(query.get("seed", ["42"])[0])
    scenario = query.get("scenario", ["medium"])[0]
    include_sensitivity = query.get("sensitivity", ["off"])[0] == "on"
    include_stress = query.get("stress", ["off"])[0] == "on"
    data_source = query.get("data_source", ["synthetic"])[0]
    trace_kind = query.get("trace_kind", ["custom"])[0]
    trace_path = query.get("trace_path", [""])[0].strip().strip('"')
    selected_names = selected_algorithm_names(query)

    started = time.time()
    trace_bundle = None
    if data_source == "real" and trace_path:
        loaders = {
            "alibaba": load_alibaba_trace,
            "bitbrains": load_bitbrains_trace,
            "custom": load_custom_real_trace,
        }
        loader = loaders.get(trace_kind, load_custom_real_trace)
        trace_bundle = loader(trace_path, max_vms=vms, traffic_profile=scenario)
        generated_vms = trace_bundle.vms
        traffic = trace_bundle.traffic
        vms = len(generated_vms)
    else:
        generated_vms = generate_vms(vms, seed=seed)
        traffic = generate_traffic_matrix(vms, seed=seed, profile=scenario)
    df, results, data_center = run_simulation(
        num_pms=pms,
        num_vms=vms,
        ants=ants,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
        evaporation=evaporation,
        seed=seed,
        vms=generated_vms,
        traffic=traffic,
        traffic_profile=scenario,
    )
    display_df = df[df["algorithm"].isin(selected_names)].copy()
    display_results = [result for result in results if result.name in selected_names]
    if display_df.empty or not display_results:
        display_df = df.copy()
        display_results = results
    best = min(display_results, key=lambda result: (result.sla_violations > 0, result.total_power, result.average_hop_count))
    display_df["best_algorithm"] = display_df["algorithm"].apply(lambda name: "YES" if name == best.name else "")
    run_id = f"run_{int(started)}_{pms}pms_{vms}vms_{ants}ants_{iterations}iter"
    output_dir = os.path.join(RESULTS_ROOT, "runs", run_id)
    save_results_csv(display_df, output_dir)
    if trace_bundle:
        pd.DataFrame([trace_bundle.profile]).to_csv(os.path.join(output_dir, "dataset_profile.csv"), index=False)
        save_processed_outputs(trace_bundle.raw_frame, output_dir)
        save_traffic_outputs(traffic, output_dir)
    sensitivity_df = run_parameter_sensitivity(generated_vms, traffic, pms, seed) if include_sensitivity else None
    stress_df = run_sla_stress(generated_vms, traffic, pms, seed) if include_stress else None
    time_window_df = run_time_window_simulation(trace_bundle, pms, scenario, seed) if trace_bundle else None
    elapsed = time.time() - started
    history_df = append_run_history(output_dir, df, query, elapsed)
    generate_all_charts(
        display_df,
        display_results,
        data_center,
        output_dir,
        traffic=traffic,
        sensitivity_df=sensitivity_df,
        stress_df=stress_df,
        history_df=history_df,
        trace_frame=trace_bundle.raw_frame if trace_bundle else None,
        time_window_df=time_window_df,
    )
    create_report(output_dir, display_df, best, run_id, query, trace_bundle=trace_bundle, time_window_df=time_window_df)
    image_base = f"/results/runs/{run_id}"
    gallery = chart_gallery_html(
        image_base,
        output_dir,
        trace_bundle=trace_bundle,
        include_sensitivity=include_sensitivity,
        include_stress=include_stress,
        time_window_df=time_window_df,
    )

    return f"""
<section>
  <div class="panel-title">
    <div>
      <h2>Recommended Algorithm</h2>
      <p class="hint">Automatically selected from the algorithms enabled in the selector. SLA-safe results are preferred before total power.</p>
    </div>
    <span class="pill">Run complete</span>
  </div>
  <div class="run-summary">
    <div class="hero-metric"><span>Recommended</span><strong>{best.name}</strong></div>
    <div class="hero-metric"><span>Total power</span><strong>{best.total_power:.0f} W</strong></div>
    <div class="hero-metric"><span>Energy savings</span><strong>{best.energy_savings_pct:.2f}%</strong></div>
    <div class="hero-metric"><span>SLA violations</span><strong>{best.sla_violations}</strong></div>
  </div>
  <p class="hint">Completed in {elapsed:.1f} seconds under the <strong>{scenario}</strong> traffic scenario. Selected algorithms: <strong>{", ".join(display_df["algorithm"].tolist())}</strong>. Results are saved in <code>{output_dir}</code>.</p>
  <div class="toolbar">
    <a class="button-link" href="{image_base}/report.html" target="_blank">Open HTML report</a>
    <a class="button-link" href="{image_base}/algorithm_comparison.csv" target="_blank">Download comparison CSV</a>
  </div>
</section>
{real_trace_summary_html(trace_bundle) if trace_bundle else ""}
<section>
  <div class="panel-title">
    <div><h2>Metric Cards</h2><p class="hint">Quick-read cards for power, utilization, hops, and SLA status.</p></div>
  </div>
  {result_cards_html(display_df)}
</section>
<section>
  <div class="panel-title">
    <div><h2>Comparison Table</h2><p class="hint">The highlighted row is the recommended algorithm for the selected run.</p></div>
    <span class="pill pill-blue">Formatted</span>
  </div>
  <div class="table-wrap">{table_html(display_df)}</div>
</section>
<section>
  <div class="panel-title">
    <div><h2>Chart Gallery</h2><p class="hint">High-resolution presentation figures generated for this run.</p></div>
    <span class="pill pill-blue">PNG gallery</span>
  </div>
  {gallery}
</section>
<section>
  <h2>VM Placement Across ToR Switches</h2>
  <p class="hint">Each cell is one physical machine. The number inside the cell is how many VMs TA-MOACO placed there.</p>
  <img src="{image_base}/topology_placement.png" alt="Topology VM placement">
</section>
<section>
  <h2>Final Recommended VM Placement</h2>
  {placement_table_html(best)}
</section>
<section>
  <h2>How TA-MOACO Works</h2>
  <ol>
    <li>Each ant builds a VM placement while checking CPU and RAM capacity.</li>
    <li>The pheromone matrix stores learned VM-to-PM placement preference.</li>
    <li>The resource heuristic packs VMs onto balanced, active servers.</li>
    <li>The traffic-affinity heuristic keeps communicating VMs close in the fat-tree network.</li>
    <li>Pheromone evaporates by {evaporation:.2f}, then elite ants and the global best reinforce better placements.</li>
    <li>The objective minimizes server power, network power, active PMs, and traffic hop count.</li>
  </ol>
</section>
"""


def controls(query: dict[str, list[str]] | None = None) -> str:
    """Render the experiment control form and input guide."""

    query = query or {}
    pms = int(query.get("pms", ["100"])[0])
    vms = int(query.get("vms", ["200"])[0])
    ants = int(query.get("ants", ["50"])[0])
    iterations = int(query.get("iterations", ["15"])[0])
    alpha = float(query.get("alpha", ["1.0"])[0])
    beta = float(query.get("beta", ["2.0"])[0])
    evaporation = float(query.get("evaporation", ["0.1"])[0])
    seed = int(query.get("seed", ["42"])[0])
    scenario = query.get("scenario", ["medium"])[0]
    sensitivity = query.get("sensitivity", ["off"])[0] == "on"
    stress = query.get("stress", ["off"])[0] == "on"
    data_source = query.get("data_source", ["synthetic"])[0]
    trace_kind = query.get("trace_kind", ["custom"])[0]
    trace_path = query.get("trace_path", [""])[0]
    estimated = estimate_seconds(pms, vms, ants, iterations)
    if sensitivity:
        estimated += 20
    if stress:
        estimated += 20
    scenario_options = "".join(
        f'<option value="{name}" {"selected" if scenario == name else ""}>{label}</option>'
        for name, label in [("low", "Low traffic"), ("medium", "Medium traffic"), ("peak", "Peak traffic")]
    )
    data_options = "".join(
        f'<option value="{value}" {"selected" if data_source == value else ""}>{label}</option>'
        for value, label in [("synthetic", "Synthetic demo data"), ("real", "Real-world trace CSV")]
    )
    trace_options = "".join(
        f'<option value="{value}" {"selected" if trace_kind == value else ""}>{label}</option>'
        for value, label in [("custom", "Generic CSV trace"), ("bitbrains", "Bitbrains-style trace"), ("alibaba", "Alibaba Cluster Trace")]
    )
    return f"""
<form id="run-form">
  <div class="panel-title">
    <div>
      <h2>Experiment Control Panel</h2>
      <p class="hint">Configure the data center scale, TA-MOACO search effort, and reproducible random seed.</p>
    </div>
    <span class="pill">Estimated {estimated}s</span>
  </div>
  <div class="run-layout">
    <div class="inputs">
      <label>PMs <input name="pms" type="number" min="20" max="300" value="{pms}"></label>
      <label>VM sample <input name="vms" type="number" min="200" max="5000" step="200" list="sample-sizes" value="{vms}"></label>
      <datalist id="sample-sizes">
        <option value="200"></option>
        <option value="400"></option>
        <option value="800"></option>
        <option value="1000"></option>
        <option value="2000"></option>
        <option value="5000"></option>
      </datalist>
      <label>Ants <input name="ants" type="number" min="10" max="80" value="{ants}"></label>
      <label>Iterations <input name="iterations" type="number" min="5" max="40" value="{iterations}"></label>
      <label>Alpha <input name="alpha" type="number" min="0.1" max="3" step="0.1" value="{alpha}"></label>
      <label>Beta <input name="beta" type="number" min="0.1" max="5" step="0.1" value="{beta}"></label>
      <label>Evaporation <input name="evaporation" type="number" min="0.01" max="0.5" step="0.01" value="{evaporation}"></label>
      <label>Seed <input name="seed" type="number" min="1" max="9999" value="{seed}"></label>
      <label>Scenario <select name="scenario">{scenario_options}</select></label>
      <label>Data <select name="data_source">{data_options}</select></label>
      <label>Trace type <select name="trace_kind">{trace_options}</select></label>
      <label>Trace CSV path <input class="path-input" name="trace_path" type="text" value="{trace_path}" placeholder="C:\\path\\trace.csv"></label>
      <input name="estimated_seconds" type="hidden" value="{estimated}">
    </div>
    <button type="submit">Run algorithms</button>
  </div>
  <div class="input-group">
    <h3>Algorithm selector</h3>
    {algorithm_selector_html(query)}
  </div>
  <div class="input-group">
    <h3>Diagnostics</h3>
    <div class="algorithm-picker">
      <label><input class="check" name="sensitivity" type="checkbox" {checked(sensitivity)}> Parameter sensitivity</label>
      <label><input class="check" name="stress" type="checkbox" {checked(stress)}> SLA stress test</label>
    </div>
  </div>
  <p class="hint">Recommended presentation setting: 100-120 PMs, 200-400 VMs, 50-60 ants, 10-15 iterations. Large real-data samples up to 5000 VMs are supported for preprocessing; algorithm runtime will grow with sample size.</p>
</form>
<section>
  <div class="panel-title">
    <div>
      <h2>Preprocessing Status</h2>
      <p class="hint">The dashboard follows this pipeline during each run and reports live progress with ETA below.</p>
    </div>
    <span class="pill pill-blue">Ready</span>
  </div>
  <div class="pipeline">
    <div class="step"><b>1. Data input</b><span>Generate synthetic VMs or read a real trace CSV preview.</span></div>
    <div class="step"><b>2. Preprocess</b><span>Normalize CPU/RAM, infer traffic, and cache processed data.</span></div>
    <div class="step"><b>3. Place VMs</b><span>Run selected baselines and TA-MOACO with capacity checks.</span></div>
    <div class="step"><b>4. Evaluate</b><span>Compute power, switches, hops, SLA, and utilization metrics.</span></div>
    <div class="step"><b>5. Export</b><span>Save CSV, PNG figures, VM placement, and HTML report.</span></div>
  </div>
</section>
{dataset_preview_html(query)}
<section>
  <div class="panel-title">
    <div>
      <h2>What The Inputs Mean</h2>
      <p class="hint">Use this section to explain the simulation setup before presenting the results.</p>
    </div>
    <span class="pill">Input guide</span>
  </div>
  <div class="mini-grid">
    <div class="feature"><strong>PMs</strong> Physical Machines are the real data center servers. Each PM has limited CPU and RAM capacity, and the algorithm decides which VMs should run on which PMs.</div>
    <div class="feature"><strong>VM sample</strong> Number of VMs to load or generate. Real traces support samples of 200, 400, 800, 1000, 2000, and 5000 VMs with chunked preprocessing.</div>
    <div class="feature"><strong>Ants</strong> Ants are candidate solution builders in ACO. More ants explore more VM placements, which can improve quality but increases runtime.</div>
    <div class="feature"><strong>Iterations</strong> Iterations are optimization rounds. After each round, pheromone is updated so later ants learn from better placements.</div>
    <div class="feature"><strong>Alpha</strong> Alpha controls how strongly ants trust pheromone memory from earlier high-quality placements.</div>
    <div class="feature"><strong>Beta</strong> Beta controls how strongly ants trust the resource and traffic-affinity heuristic during placement.</div>
    <div class="feature"><strong>Evaporation</strong> Evaporation removes old pheromone. Higher values adapt faster, while lower values preserve learned placement patterns longer.</div>
    <div class="feature"><strong>Seed</strong> Seed controls randomness. Using the same seed gives the same synthetic workload and makes results reproducible for presentation.</div>
    <div class="feature"><strong>Scenario</strong> Scenario controls traffic intensity. Low, medium, and peak traffic show how TA-MOACO behaves under different network communication loads.</div>
    <div class="feature"><strong>Data Source</strong> Synthetic mode generates demo workloads. Real-world trace mode reads a CSV file and converts trace rows into VM demands.</div>
    <div class="feature"><strong>Trace CSV Path</strong> For real-world mode, enter a local CSV path. Recommended columns are <code>vm_id,cpu,ram,app_id,timestamp</code>.</div>
  </div>
  <details class="explain">
    <summary>Parameter explanation for faculty Q&amp;A</summary>
    <p class="hint"><strong>Alpha</strong> controls pheromone memory, <strong>beta</strong> controls heuristic pressure, and <strong>evaporation</strong> decides how quickly old placement memory fades. Increasing ants or iterations usually improves search quality but increases runtime.</p>
  </details>
</section>
<section id="progress-box" class="progress-wrap">
  <h2>Simulation Running</h2>
  <div class="progress-track"><div id="progress-fill" class="progress-fill"></div></div>
  <div class="status-row">
    <span>Status: <strong id="status-text">Waiting...</strong></span>
    <span>Progress: <strong id="percent">0%</strong></span>
    <span>Elapsed: <strong id="elapsed">0:00</strong></span>
    <span>Estimated remaining: <strong id="eta">{estimated}</strong></span>
  </div>
</section>
<div id="results"></div>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/results/"):
            self.serve_file(parsed.path.lstrip("/"))
            return

        if parsed.path == "/api/run":
            self.serve_api_run(parse_qs(parsed.query))
            return

        query = parse_qs(parsed.query)
        body = controls(query) + """
<section>
  <div class="panel-title">
    <div>
      <h2>Presentation Flow</h2>
      <p class="hint">Use this order when explaining the system to your teacher.</p>
    </div>
  </div>
  <div class="mini-grid">
    <div class="feature"><strong>1. Generate Workload</strong> The simulator creates PM capacity, VM CPU/RAM demand, and clustered VM traffic.</div>
    <div class="feature"><strong>2. Run Four Algorithms</strong> FFD, ACS-VMP, Traffic-Aware VMP, and TA-MOACO are compared on the same workload.</div>
    <div class="feature"><strong>3. Explain Tradeoffs</strong> Show how traffic-only placement can lower hops but waste server power.</div>
    <div class="feature"><strong>4. Defend TA-MOACO</strong> TA-MOACO balances consolidation and traffic locality, reducing total energy.</div>
  </div>
</section>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html_page(body))

    def serve_api_run(self, query: dict[str, list[str]]) -> None:
        try:
            html = run_simulation_html(query)
            payload = {"ok": True, "html": html}
        except Exception as exc:
            payload = {"ok": False, "error": repr(exc)}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_file(self, relative_path: str) -> None:
        clean_path = relative_path.split("?", 1)[0]
        if clean_path.startswith("results/"):
            clean_path = clean_path[len("results/") :]
        path = os.path.abspath(os.path.join(RESULTS_ROOT, clean_path))
        if not path.startswith(RESULTS_ROOT) or not os.path.exists(path):
            self.send_error(404)
            return
        self.send_response(200)
        if path.endswith(".png"):
            content_type = "image/png"
        elif path.endswith(".html"):
            content_type = "text/html; charset=utf-8"
        elif path.endswith(".csv"):
            content_type = "text/csv; charset=utf-8"
        else:
            content_type = "text/plain; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.end_headers()
        with open(path, "rb") as file:
            self.wfile.write(file.read())


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", PORT), DashboardHandler)
    print(f"Dashboard running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()
