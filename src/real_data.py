from __future__ import annotations

import os
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from models import TrafficMatrix, VirtualMachine


CACHE_DIR = os.path.join("results", "cache")
CACHE_VERSION = "traffic-v2"


@dataclass
class RealTraceBundle:
    """Normalized real-world trace data used by the simulator."""

    vms: List[VirtualMachine]
    traffic: TrafficMatrix
    raw_frame: pd.DataFrame
    profile: Dict[str, float | int | str]
    source_name: str
    traffic_mode: str
    warnings: List[str] = field(default_factory=list)


COLUMN_ALIASES = {
    "vm_id": ["vm_id", "vm", "id", "container_id", "task_id", "instance_id", "instance_name"],
    "cpu": ["cpu", "cpu_request", "cpu_usage", "avg_cpu", "mean_cpu", "cpu_util", "cpu_cores", "cpu_util_percent"],
    "ram": ["ram", "memory", "mem", "mem_usage", "memory_usage", "avg_mem", "mean_mem", "ram_gb", "mem_gb"],
    "timestamp": ["timestamp", "time", "start_time", "sample_time", "ts", "time_stamp"],
    "network_in": ["network_in", "net_in", "bytes_in", "rx", "rx_bytes", "net_rx"],
    "network_out": ["network_out", "net_out", "bytes_out", "tx", "tx_bytes", "net_tx"],
    "app_id": ["app_id", "job_id", "service_id", "collection_id", "user_id", "application", "app"],
    "machine_id": ["machine_id", "machine", "host_id", "server_id", "pm_id", "physical_machine"],
}


def _find_column(frame: pd.DataFrame, canonical: str) -> str | None:
    """Return the matching column name for a canonical field."""

    lower_to_original = {column.lower().strip(): column for column in frame.columns}
    for alias in COLUMN_ALIASES[canonical]:
        if alias in lower_to_original:
            return lower_to_original[alias]
    return None


def _detect_columns_from_names(columns) -> Dict[str, str | None]:
    header_frame = pd.DataFrame(columns=columns)
    return detect_columns(header_frame)


def _cache_key(csv_file, max_rows: int, traffic_profile: str, source_name: str) -> str:
    """Build a stable cache key from file metadata and preprocessing options."""

    path = os.path.abspath(str(csv_file))
    stat = os.stat(path)
    raw = f"{CACHE_VERSION}|{path}|{stat.st_size}|{stat.st_mtime_ns}|{max_rows}|{traffic_profile}|{source_name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _read_csv_sample(csv_file, max_rows: int, chunk_size: int = 50_000) -> tuple[pd.DataFrame, bool, Dict[str, str | None], Dict[str, float | int | str]]:
    """Read up to max_rows using chunks and only columns relevant to preprocessing."""

    header = pd.read_csv(csv_file, nrows=0)
    detected = _detect_columns_from_names(header.columns)
    usecols = sorted({column for column in detected.values() if column is not None})
    if not usecols and len(header.columns) > 0:
        usecols = [header.columns[0]]

    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(csv_file, chunksize=chunk_size, usecols=usecols or None):
        chunks.append(chunk)
        total_rows += len(chunk)
        if total_rows >= max_rows:
            break
    if chunks:
        was_sampled_or_multi_chunk = len(chunks) > 1 or total_rows >= max_rows
        frame = pd.concat(chunks, ignore_index=True).head(max_rows)
    else:
        frame = pd.read_csv(csv_file, nrows=max_rows, usecols=usecols or None)
        was_sampled_or_multi_chunk = False
    stats = {
        "rows_loaded": int(len(frame)),
        "chunks_read": int(len(chunks)),
        "chunk_size": int(chunk_size),
        "selected_columns": ", ".join(usecols) if usecols else "all",
        "memory_mb": round(float(frame.memory_usage(deep=True).sum()) / (1024 * 1024), 3),
    }
    return frame, was_sampled_or_multi_chunk, detected, stats


def _scale_cpu(series: pd.Series, warnings: List[str]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        warnings.append("CPU column existed but contained no numeric values; default CPU demand was used.")
        return pd.Series([8.0] * len(series))
    values = values.fillna(values.median())
    if values.max() <= 1.0:
        values = values * 100.0
    elif values.max() > 100.0:
        values = values / values.max() * 80.0
    return values.clip(lower=1.0, upper=95.0).round(2)


def _scale_ram(series: pd.Series, warnings: List[str]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        warnings.append("RAM column existed but contained no numeric values; RAM was estimated from CPU.")
        return values
    values = values.fillna(values.median())
    if values.max() <= 1.0:
        values = values * 256.0
    elif values.max() > 256.0:
        values = values / values.max() * 220.0
    return values.clip(lower=1.0, upper=245.0).round(2)


def detect_columns(frame: pd.DataFrame) -> Dict[str, str | None]:
    """Detect supported real-world trace columns."""

    return {canonical: _find_column(frame, canonical) for canonical in COLUMN_ALIASES}


def normalize_custom_trace(frame: pd.DataFrame, max_vms: int = 800, detected: Dict[str, str | None] | None = None) -> tuple[pd.DataFrame, List[str], Dict[str, str | None]]:
    """Normalize a CSV frame into simulator columns without crashing on missing fields."""

    warnings: List[str] = []
    detected = detected or detect_columns(frame)

    normalized = pd.DataFrame()
    normalized["vm_id"] = frame[detected["vm_id"]] if detected["vm_id"] else range(len(frame))
    if detected["vm_id"] is None:
        warnings.append("Missing vm_id column; generated sequential VM IDs.")

    if detected["cpu"]:
        normalized["cpu"] = _scale_cpu(frame[detected["cpu"]], warnings)
    else:
        normalized["cpu"] = 8.0
        warnings.append("Missing CPU column; default CPU demand of 8 was used.")

    if detected["ram"]:
        normalized["ram"] = _scale_ram(frame[detected["ram"]], warnings)
    else:
        normalized["ram"] = (normalized["cpu"] * 2.5).clip(lower=4.0, upper=245.0).round(2)
        warnings.append("Missing RAM/memory column; RAM was estimated from CPU demand.")

    if normalized["ram"].isna().any():
        normalized["ram"] = (normalized["cpu"] * 2.5).clip(lower=4.0, upper=245.0).round(2)

    if detected["app_id"]:
        normalized["app_id"] = frame[detected["app_id"]].astype(str).fillna("unknown-app")
    elif detected["machine_id"]:
        normalized["app_id"] = "machine-" + frame[detected["machine_id"]].astype(str).fillna("unknown")
        warnings.append("Missing app_id column; machine_id was used to infer traffic groups.")
    else:
        normalized["app_id"] = "app-" + (pd.Series(range(len(frame))) // 10).astype(str)
        warnings.append("Missing app_id and machine_id columns; generated synthetic app groups for traffic inference.")

    if detected["timestamp"]:
        normalized["timestamp"] = pd.to_numeric(frame[detected["timestamp"]], errors="coerce").fillna(0)
    else:
        normalized["timestamp"] = 0
        warnings.append("Missing timestamp column; all VMs assigned to time window 0.")

    normalized["network_in"] = pd.to_numeric(frame[detected["network_in"]], errors="coerce").fillna(0) if detected["network_in"] else 0.0
    normalized["network_out"] = pd.to_numeric(frame[detected["network_out"]], errors="coerce").fillna(0) if detected["network_out"] else 0.0
    if detected["network_in"] is None and detected["network_out"] is None:
        warnings.append("Missing network_in/network_out columns; traffic will be inferred later.")

    normalized["machine_id"] = frame[detected["machine_id"]].astype(str) if detected["machine_id"] else "unknown"
    normalized = normalized.dropna(subset=["cpu", "ram"]).sort_values(["timestamp", "vm_id"]).head(max_vms).reset_index(drop=True)
    normalized["sim_vm_id"] = range(len(normalized))
    if normalized.empty:
        warnings.append("No valid rows remained after normalization.")
    return normalized, warnings, detected


def infer_traffic_from_apps(frame: pd.DataFrame, profile: str = "medium") -> TrafficMatrix:
    """Infer normalized VM-to-VM traffic from workload similarity.

    Public traces often have resource usage but no complete VM-to-VM flow
    matrix. The inference below treats traffic affinity as a correlation score:
    same application, close timestamps, similar CPU/RAM demand, similar network
    usage, and a tiny deterministic noise term. Scores are clipped to 0..1 so
    the generated matrix is comparable across datasets and traffic profiles.
    """

    profile_weights = {
        "low": 0.65,
        "medium": 1.0,
        "peak": 1.25,
    }
    intensity = profile_weights.get(profile, profile_weights["medium"])
    traffic: TrafficMatrix = {}
    if len(frame) <= 800:
        rows = frame[["sim_vm_id", "app_id", "timestamp", "cpu", "ram", "network_in", "network_out"]].to_dict("records")
        for index, src in enumerate(rows):
            for dst in rows[index + 1 :]:
                demand = _pair_demand(src, dst, intensity)
                if demand > 0:
                    traffic[(int(src["sim_vm_id"]), int(dst["sim_vm_id"]))] = round(demand, 4)
        return traffic

    # Large traces use bounded local neighborhoods to avoid O(n^2) memory growth.
    for _app_id, group in frame.sort_values("sim_vm_id").groupby("app_id"):
        rows = group[["sim_vm_id", "app_id", "timestamp", "cpu", "ram", "network_in", "network_out"]].to_dict("records")
        for index, src in enumerate(rows):
            for dst in rows[index + 1 : index + 9]:
                demand = _pair_demand(src, dst, intensity)
                if demand > 0:
                    traffic[(int(src["sim_vm_id"]), int(dst["sim_vm_id"]))] = round(demand, 4)

    for _timestamp, group in frame.sort_values("sim_vm_id").groupby("timestamp"):
        rows = group[["sim_vm_id", "app_id", "timestamp", "cpu", "ram", "network_in", "network_out"]].to_dict("records")
        for index, src in enumerate(rows):
            for dst in rows[index + 1 : index + 4]:
                key = (int(src["sim_vm_id"]), int(dst["sim_vm_id"]))
                if key not in traffic:
                    demand = _pair_demand(src, dst, intensity)
                    if demand > 0:
                        traffic[key] = round(demand, 4)
    return traffic


def _similarity(a: float, b: float, scale: float) -> float:
    """Return 1 for equal values and approach 0 as values diverge."""

    return max(0.0, 1.0 - abs(float(a) - float(b)) / max(scale, 1.0))


def _pair_demand(src, dst, intensity: float) -> float:
    # Same app and close timestamps are the strongest signal because services
    # commonly communicate internally and within the same scheduling interval.
    same_app_score = 1.0 if src["app_id"] == dst["app_id"] else 0.0
    timestamp_gap = abs(float(src["timestamp"]) - float(dst["timestamp"]))
    same_timestamp_score = 1.0 if timestamp_gap == 0 else 0.0
    nearby_timestamp_score = max(0.0, 1.0 - timestamp_gap / 3.0)

    # Similar CPU/RAM/network behavior suggests VMs may belong to the same tier
    # or workload phase. These are softer signals than app/timestamp.
    cpu_score = _similarity(src["cpu"], dst["cpu"], 100.0)
    ram_score = _similarity(src["ram"], dst["ram"], 256.0)
    src_network = float(src["network_in"]) + float(src["network_out"])
    dst_network = float(dst["network_in"]) + float(dst["network_out"])
    network_score = _similarity(src_network, dst_network, max(src_network, dst_network, 1.0))

    correlation = (
        0.28 * same_app_score
        + 0.18 * same_timestamp_score
        + 0.13 * nearby_timestamp_score
        + 0.15 * cpu_score
        + 0.14 * ram_score
        + 0.12 * network_score
    )

    # Deterministic noise breaks ties without making repeated runs unstable.
    noise_seed = int(src["sim_vm_id"]) * 1_000_003 + int(dst["sim_vm_id"]) * 97
    noise = random.Random(noise_seed).uniform(-0.025, 0.025)
    return min(1.0, max(0.0, correlation * intensity + noise))


def profile_dataset(
    frame: pd.DataFrame,
    traffic: TrafficMatrix,
    source_name: str,
    traffic_mode: str,
    warnings: List[str] | None = None,
    detected_columns: Dict[str, str | None] | None = None,
    loading_stats: Dict[str, float | int | str] | None = None,
) -> Dict[str, float | int | str]:
    """Return dataset quality and resource statistics for reports and CSV export."""

    warnings = warnings or []
    detected_columns = detected_columns or {}
    loading_stats = loading_stats or {}
    profile = {
        "source_name": source_name,
        "traffic_mode": traffic_mode,
        "vm_count": int(len(frame)),
        "app_count": int(frame["app_id"].nunique()) if not frame.empty else 0,
        "machine_count": int(frame["machine_id"].nunique()) if "machine_id" in frame else 0,
        "time_windows": int(frame["timestamp"].nunique()) if not frame.empty else 0,
        "missing_values": int(frame.isna().sum().sum()),
        "cpu_min": round(float(frame["cpu"].min()), 2) if not frame.empty else 0.0,
        "avg_cpu": round(float(frame["cpu"].mean()), 2) if not frame.empty else 0.0,
        "cpu_median": round(float(frame["cpu"].median()), 2) if not frame.empty else 0.0,
        "peak_cpu": round(float(frame["cpu"].max()), 2) if not frame.empty else 0.0,
        "ram_min": round(float(frame["ram"].min()), 2) if not frame.empty else 0.0,
        "avg_ram": round(float(frame["ram"].mean()), 2) if not frame.empty else 0.0,
        "ram_median": round(float(frame["ram"].median()), 2) if not frame.empty else 0.0,
        "peak_ram": round(float(frame["ram"].max()), 2) if not frame.empty else 0.0,
        "traffic_edges": int(len(traffic)),
        "avg_traffic": round(float(sum(traffic.values()) / max(1, len(traffic))), 4),
        "warnings": " | ".join(warnings) if warnings else "none",
    }
    profile.update(traffic_statistics(traffic))
    profile.update(loading_stats)
    for canonical, actual in detected_columns.items():
        profile[f"column_{canonical}"] = actual or "missing"
    return profile


def save_dataset_profile(profile: Dict[str, float | int | str], output_dir: str = "results") -> str:
    """Save a single-row dataset profile CSV."""

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dataset_profile.csv")
    pd.DataFrame([profile]).to_csv(path, index=False)
    return path


def save_processed_outputs(frame: pd.DataFrame, output_dir: str = "results") -> tuple[str, str]:
    """Save normalized VM rows and simulator-ready VM columns."""

    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, "processed_dataset.csv")
    vms_path = os.path.join(output_dir, "processed_vms.csv")
    frame.to_csv(dataset_path, index=False)
    frame[["sim_vm_id", "cpu", "ram"]].rename(columns={"sim_vm_id": "vm_id"}).to_csv(vms_path, index=False)
    return vms_path, dataset_path


def traffic_to_dataframe(traffic: TrafficMatrix) -> pd.DataFrame:
    """Convert the sparse traffic dictionary to a CSV-friendly edge table."""

    return pd.DataFrame(
        [{"src_vm": src, "dst_vm": dst, "traffic": demand} for (src, dst), demand in sorted(traffic.items())]
    )


def traffic_statistics(traffic: TrafficMatrix) -> Dict[str, float | int]:
    """Return summary statistics for normalized inferred traffic values."""

    if not traffic:
        return {
            "traffic_edges": 0,
            "traffic_min": 0.0,
            "traffic_mean": 0.0,
            "traffic_median": 0.0,
            "traffic_max": 0.0,
            "traffic_density_edges": 0,
        }
    values = pd.Series(list(traffic.values()), dtype=float)
    return {
        "traffic_edges": int(len(values)),
        "traffic_min": round(float(values.min()), 4),
        "traffic_mean": round(float(values.mean()), 4),
        "traffic_median": round(float(values.median()), 4),
        "traffic_max": round(float(values.max()), 4),
        "traffic_density_edges": int((values > 0).sum()),
    }


def save_traffic_outputs(traffic: TrafficMatrix, output_dir: str = "results") -> tuple[str, str]:
    """Save inferred traffic matrix and statistics as CSV files."""

    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "traffic_matrix.csv")
    stats_path = os.path.join(output_dir, "traffic_statistics.csv")
    traffic_to_dataframe(traffic).to_csv(matrix_path, index=False)
    pd.DataFrame([traffic_statistics(traffic)]).to_csv(stats_path, index=False)
    return matrix_path, stats_path


def load_custom_real_trace(
    csv_file,
    max_vms: int = 800,
    traffic_profile: str = "medium",
    source_name: str = "Custom CSV",
    chunk_size: int = 50_000,
) -> RealTraceBundle:
    """Load a generic CSV trace with tolerant column detection and chunking."""

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(csv_file, max_vms, traffic_profile, source_name)
    cached_dataset = os.path.join(CACHE_DIR, f"{key}_processed_dataset.csv")
    cached_profile = os.path.join(CACHE_DIR, f"{key}_profile.json")
    warnings: List[str] = []

    if os.path.exists(cached_dataset) and os.path.exists(cached_profile):
        normalized = pd.read_csv(cached_dataset)
        with open(cached_profile, "r", encoding="utf-8") as file:
            profile = json.load(file)
        warnings = [warning for warning in str(profile.get("warnings", "")).split(" | ") if warning and warning != "none"]
        profile["cache_status"] = "hit"
        traffic = infer_traffic_from_apps(normalized, profile=traffic_profile)
        save_dataset_profile(profile)
        save_processed_outputs(normalized)
        save_traffic_outputs(traffic)
        vms = [
            VirtualMachine(id=int(row.sim_vm_id), cpu=float(row.cpu), ram=float(row.ram))
            for row in normalized.itertuples()
        ]
        return RealTraceBundle(vms=vms, traffic=traffic, raw_frame=normalized, profile=profile, source_name=source_name, traffic_mode="inferred", warnings=warnings)

    frame, used_chunks, detected, loading_stats = _read_csv_sample(csv_file, max_rows=max_vms, chunk_size=chunk_size)
    normalized, warnings, detected = normalize_custom_trace(frame, max_vms=max_vms, detected=detected)
    if used_chunks:
        warnings.append(f"CSV loaded with chunked reading; sampled first {len(normalized)} normalized rows.")
    traffic = infer_traffic_from_apps(normalized, profile=traffic_profile)
    vms = [
        VirtualMachine(id=int(row.sim_vm_id), cpu=float(row.cpu), ram=float(row.ram))
        for row in normalized.itertuples()
    ]
    profile = profile_dataset(
        normalized,
        traffic,
        source_name=source_name,
        traffic_mode="inferred from app_id/timestamp/network hints",
        warnings=warnings,
        detected_columns=detected,
        loading_stats={**loading_stats, "cache_status": "miss"},
    )
    normalized.to_csv(cached_dataset, index=False)
    with open(cached_profile, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)
    save_dataset_profile(profile)
    save_processed_outputs(normalized)
    save_traffic_outputs(traffic)
    return RealTraceBundle(
        vms=vms,
        traffic=traffic,
        raw_frame=normalized,
        profile=profile,
        source_name=source_name,
        traffic_mode="inferred",
        warnings=warnings,
    )


def load_bitbrains_trace(csv_file, max_vms: int = 800, traffic_profile: str = "medium") -> RealTraceBundle:
    """Load a Bitbrains-style VM trace."""

    return load_custom_real_trace(csv_file, max_vms=max_vms, traffic_profile=traffic_profile, source_name="Bitbrains-style trace")


def load_alibaba_trace(csv_file, max_vms: int = 800, traffic_profile: str = "medium") -> RealTraceBundle:
    """Load an Alibaba Cluster Trace-style CSV using the generic normalizer."""

    return load_custom_real_trace(csv_file, max_vms=max_vms, traffic_profile=traffic_profile, source_name="Alibaba Cluster Trace-style CSV")
