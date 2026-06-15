from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from flask import Flask, Response, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Vercel serverless functions can write to /tmp, not to the deployed project
# directory. Local tests use results/vercel_tmp so the normal local dashboard
# continues to use results/.
if os.environ.get("VERCEL"):
    os.environ.setdefault("TA_MOACO_RESULTS_DIR", "/tmp/results")
else:
    os.environ.setdefault("TA_MOACO_RESULTS_DIR", str(ROOT / "results" / "vercel_tmp"))

from local_dashboard import RESULTS_ROOT, controls, html_page, run_simulation_html


app = Flask(__name__)


def _query_dict() -> dict[str, list[str]]:
    """Convert Flask query args into the local dashboard's expected shape."""

    return {key: request.args.getlist(key) for key in request.args.keys()}


def _presentation_flow() -> str:
    """Return the same overview section shown by the local dashboard server."""

    return """
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


@app.get("/")
def home() -> Response:
    body = controls(_query_dict()) + _presentation_flow()
    return Response(html_page(body), mimetype="text/html")


@app.get("/api/run")
def api_run() -> Response:
    try:
        html = run_simulation_html(_query_dict())
        payload = {"ok": True, "html": html}
    except Exception as exc:
        payload = {"ok": False, "error": repr(exc)}
    return Response(json.dumps(payload), mimetype="application/json")


@app.get("/results/<path:relative_path>")
def results_file(relative_path: str) -> Response:
    return send_from_directory(RESULTS_ROOT, relative_path)


@app.get("/api/health")
def health() -> Response:
    return Response(json.dumps({"ok": True, "service": "ta-moaco-local-dashboard"}), mimetype="application/json")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response("", status=204)
