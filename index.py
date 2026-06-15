from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation import generate_traffic_matrix, generate_vms, run_simulation, save_results_csv
from visualization import generate_all_charts


app = Flask(__name__)


def _page(body: str) -> str:
    """Return a compact Vercel-compatible dashboard page."""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TA-MOACO VM Placement</title>
  <style>
    :root {{
      --ink:#172033; --muted:#637083; --line:#d9e2ec; --navy:#0f2742;
      --blue:#246bfe; --green:#16845f; --paper:#ffffff;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; font-family:Segoe UI,Aptos,Trebuchet MS,sans-serif; color:var(--ink);
      background:linear-gradient(135deg, rgba(36,107,254,.07), transparent 34%),
        repeating-linear-gradient(90deg, rgba(15,39,66,.045) 0 1px, transparent 1px 84px),
        #f6f9fd;
    }}
    header {{ background:linear-gradient(135deg,#0f2742,#1b5c84); color:white; padding:34px 42px; }}
    header div, main {{ max-width:1220px; margin:auto; }}
    h1 {{ margin:0 0 10px; font-size:46px; line-height:1.04; }}
    h2 {{ margin:0 0 16px; }}
    p {{ color:var(--muted); }}
    header p {{ color:#e8f4ff; font-size:18px; }}
    main {{ padding:28px 22px; }}
    section, form {{
      background:rgba(255,255,255,.94); border:1px solid var(--line); border-radius:14px;
      padding:24px; margin-bottom:22px; box-shadow:0 14px 40px rgba(35,61,93,.10);
    }}
    label {{ display:inline-flex; align-items:center; gap:8px; margin:8px 18px 8px 0; font-weight:700; }}
    input, select {{ padding:10px 12px; width:110px; border:1px solid #b7c7d7; border-radius:8px; font-size:15px; }}
    button, .button {{
      display:inline-block; background:linear-gradient(135deg,var(--blue),#00a6c8); color:white;
      border:0; border-radius:9px; padding:12px 16px; font-weight:800; text-decoration:none; cursor:pointer;
    }}
    table {{ border-collapse:collapse; width:100%; font-size:14px; }}
    th, td {{ border:1px solid #d1d5db; padding:9px 10px; text-align:right; }}
    th:first-child, td:first-child {{ text-align:left; }}
    th {{ background:#edf4fb; }}
    tr.best td {{ background:#e8fff5; font-weight:800; }}
    .cards, .gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
    .card {{ border:1px solid var(--line); border-radius:12px; padding:16px; background:#fff; }}
    .card span {{ display:block; color:var(--muted); font-size:13px; font-weight:800; text-transform:uppercase; }}
    .card strong {{ display:block; margin-top:8px; font-size:25px; color:var(--navy); }}
    img {{ width:100%; border:1px solid var(--line); border-radius:12px; background:white; }}
    .hint {{ color:var(--muted); }}
    .table-wrap {{ overflow:auto; }}
    @media (max-width:720px) {{ h1{{font-size:34px}} label{{width:100%; justify-content:space-between}} }}
  </style>
</head>
<body>
  <header><div><h1>TA-MOACO VM Placement Dashboard</h1><p>Vercel demo build for server power, network power, traffic locality, and VM placement.</p></div></header>
  <main>{body}</main>
</body>
</html>"""


def _img_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _runtime_output_dir() -> Path:
    """Return a writable output directory for Vercel or local tests."""

    if os.environ.get("VERCEL"):
        base = Path("/tmp")
    else:
        base = ROOT / "results" / "vercel_tmp"
    path = base / f"run_{int(time.time() * 1000)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


@app.get("/")
def home() -> Response:
    body = """
<form action="/run" method="get">
  <h2>Run Synthetic Simulation</h2>
  <p class="hint">This Vercel version runs synthetic workloads. Use <code>python local_dashboard.py</code> for local real-CSV file uploads and full report files.</p>
  <label>PMs <input name="pms" type="number" min="20" max="200" value="100"></label>
  <label>VMs <input name="vms" type="number" min="50" max="400" value="200"></label>
  <label>Ants <input name="ants" type="number" min="5" max="50" value="10"></label>
  <label>Iterations <input name="iterations" type="number" min="3" max="20" value="5"></label>
  <label>Seed <input name="seed" type="number" min="1" max="9999" value="42"></label>
  <label>Scenario
    <select name="scenario">
      <option value="low">Low</option>
      <option value="medium" selected>Medium</option>
      <option value="peak">Peak</option>
    </select>
  </label>
  <button type="submit">Run algorithms</button>
</form>
<section>
  <h2>What This Demonstrates</h2>
  <p>TA-MOACO balances server consolidation and traffic locality. It compares FFD, ACS-VMP, Traffic-Aware VMP, and TA-MOACO using active PMs, active switches, server power, network power, total power, hop count, utilization, and SLA violations.</p>
</section>
"""
    return Response(_page(body), mimetype="text/html")


@app.get("/run")
def run() -> Response:
    started = time.time()
    pms = max(20, min(200, int(request.args.get("pms", 100))))
    vms = max(50, min(400, int(request.args.get("vms", 200))))
    ants = max(5, min(50, int(request.args.get("ants", 10))))
    iterations = max(3, min(20, int(request.args.get("iterations", 5))))
    seed = int(request.args.get("seed", 42))
    scenario = request.args.get("scenario", "medium")

    generated_vms = generate_vms(vms, seed=seed)
    traffic = generate_traffic_matrix(vms, seed=seed, profile=scenario)
    df, results, data_center = run_simulation(
        num_pms=pms,
        num_vms=vms,
        seed=seed,
        ants=ants,
        iterations=iterations,
        vms=generated_vms,
        traffic=traffic,
        traffic_profile=scenario,
    )

    run_dir = _runtime_output_dir()
    save_results_csv(df, str(run_dir))
    chart_paths = generate_all_charts(df, results, data_center, str(run_dir), traffic=traffic)
    best = min(results, key=lambda result: (result.sla_violations > 0, result.total_power, result.average_hop_count))

    display_df = df.copy()
    display_df["best_algorithm"] = display_df["best_algorithm"].replace({"YES": "Best"})
    table_html = display_df.to_html(index=False, border=0).replace("<tr>\n      <td>Best</td>", '<tr class="best">\n      <td>Best</td>')

    cards = f"""
<div class="cards">
  <div class="card"><span>Recommended</span><strong>{best.name}</strong></div>
  <div class="card"><span>Total Power</span><strong>{best.total_power:.0f} W</strong></div>
  <div class="card"><span>Energy Savings</span><strong>{best.energy_savings_pct:.2f}%</strong></div>
  <div class="card"><span>Active Switches</span><strong>{best.active_switches}</strong></div>
</div>
"""
    figures = []
    for path in chart_paths[:8]:
        p = Path(path)
        if p.exists() and p.suffix == ".png":
            figures.append(f'<div><img src="{_img_data_uri(p)}" alt="{p.stem}"><p class="hint">{p.stem.replace("_", " ").title()}</p></div>')

    body = f"""
<section>
  <a class="button" href="/">Back</a>
  <h2>Recommended Algorithm</h2>
  {cards}
  <p class="hint">Completed in {time.time() - started:.1f}s. Parameters: {pms} PMs, {vms} VMs, {ants} ants, {iterations} iterations, {scenario} traffic.</p>
</section>
<section>
  <h2>Comparison Table</h2>
  <div class="table-wrap">{table_html}</div>
</section>
<section>
  <h2>Chart Gallery</h2>
  <div class="gallery">{''.join(figures)}</div>
</section>
<section>
  <h2>Deployment Note</h2>
  <p class="hint">Vercel serverless functions are best for a public synthetic demo. The local dashboard remains the full faculty version for real local CSV paths, persistent exported files, and large runs.</p>
</section>
"""
    return Response(_page(body), mimetype="text/html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "service": "ta-moaco-vercel"})


@app.get("/favicon.ico")
def favicon():
    return Response("", status=204)


@app.get("/README.md")
def readme():
    return send_from_directory(ROOT, "README.md")
