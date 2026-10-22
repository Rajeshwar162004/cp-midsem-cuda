"""
ncu_parser.py
============================================================
Parses Nsight Compute output (CSV rows from ncu_runner) into a
normalized, architecture-agnostic profile data model.

Responsibilities:
  - Map raw NCU metric names to our normalized field names
  - Handle multiple NCU metric naming schemes (varies by NCU version / arch)
  - Produce a NcuProfile object per kernel
  - Gracefully handle missing metrics (not all metrics exist on all GPUs)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Normalized data model ────────────────────────────────────

@dataclass
class GlobalMemoryMetrics:
    """Normalized global memory statistics."""
    # Sectors: one sector = 32 bytes.  Higher sectors/request = inefficiency
    load_sectors:    float = 0.0   # l1tex__t_sectors_pipe_lsu_mem_global_op_ld
    store_sectors:   float = 0.0   # l1tex__t_sectors_pipe_lsu_mem_global_op_st
    load_requests:   float = 0.0   # l1tex__t_requests_pipe_lsu_mem_global_op_ld
    store_requests:  float = 0.0   # l1tex__t_requests_pipe_lsu_mem_global_op_st
    # Derived
    load_sector_ratio:  float = 0.0   # sectors / requests (ideal = 1)
    store_sector_ratio: float = 0.0

    def compute_ratios(self):
        if self.load_requests > 0:
            self.load_sector_ratio = self.load_sectors / self.load_requests
        if self.store_requests > 0:
            self.store_sector_ratio = self.store_sectors / self.store_requests


@dataclass
class SharedMemoryMetrics:
    """Normalized shared memory bank-conflict statistics."""
    bank_conflicts_ld: float = 0.0   # l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
    bank_conflicts_st: float = 0.0   # l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st

    @property
    def total_conflicts(self) -> float:
        return self.bank_conflicts_ld + self.bank_conflicts_st


@dataclass
class LocalMemoryMetrics:
    """Normalized local memory (spill) traffic."""
    load_sectors:  float = 0.0   # l1tex__t_sectors_pipe_lsu_mem_local_op_ld
    store_sectors: float = 0.0   # l1tex__t_sectors_pipe_lsu_mem_local_op_st

    @property
    def has_traffic(self) -> bool:
        return self.load_sectors > 0 or self.store_sectors > 0


@dataclass
class NcuProfile:
    """Full normalized profile for one kernel invocation."""
    kernel_name:     str = ""
    duration_us:     float = 0.0    # kernel duration in microseconds
    global_mem:      GlobalMemoryMetrics  = field(default_factory=GlobalMemoryMetrics)
    shared_mem:      SharedMemoryMetrics  = field(default_factory=SharedMemoryMetrics)
    local_mem:       LocalMemoryMetrics   = field(default_factory=LocalMemoryMetrics)
    # Raw metrics preserved for debugging
    raw_metrics:     dict = field(default_factory=dict)
    # Which metrics were not found (for transparency)
    missing_metrics: list = field(default_factory=list)


# ── Metric name mappings ─────────────────────────────────────
# Maps our internal field names to possible NCU metric names.
# NCU renames metrics across versions; we try multiple names.

METRIC_MAP: dict[str, list[str]] = {
    # Global memory load sectors
    "global_load_sectors": [
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg",
    ],
    # Global memory store sectors
    "global_store_sectors": [
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.avg",
    ],
    # Global memory load requests
    "global_load_requests": [
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.avg",
    ],
    # Global memory store requests
    "global_store_requests": [
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.avg",
    ],
    # Local memory (spill) loads
    "local_load_sectors": [
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.avg",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum.per_second",
    ],
    # Local memory (spill) stores
    "local_store_sectors": [
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.avg",
    ],
    # Shared memory bank conflicts (load)
    "smem_bank_conflict_ld": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.avg",
        "shared_load_transactions_per_request",  # older metric name
    ],
    # Shared memory bank conflicts (store)
    "smem_bank_conflict_st": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.avg",
        "shared_store_transactions_per_request",
    ],
    # Duration
    "duration_us": [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.avg",           # fallback (cycles, not us)
        "Duration",                          # NCU display name
    ],
}


# ── Parser ───────────────────────────────────────────────────

def parse_ncu_rows(kernel_rows: list) -> list[NcuProfile]:
    """
    Convert a list of NcuKernelRow objects (from ncu_runner) into
    normalized NcuProfile objects.

    Args:
        kernel_rows: list of NcuKernelRow from ncu_runner.run_ncu()

    Returns:
        list of NcuProfile, one per unique kernel (averaged if multiple invocations)
    """
    # Group rows by kernel name. NCU emits one row per metric and may emit
    # multiple launches of the same kernel; its per-row values are already
    # collected values for that kernel name in ncu_runner.
    kernel_groups: dict[str, list] = {}
    for row in kernel_rows:
        name = _clean_kernel_name(row.kernel_name)
        kernel_groups.setdefault(name, []).append(row)

    profiles: list[NcuProfile] = []
    for kernel_name, rows in kernel_groups.items():
        profile = _build_profile(kernel_name, rows)
        profiles.append(profile)
        logger.debug(
            f"[{kernel_name}] "
            f"global_ld_sectors={profile.global_mem.load_sectors:.0f} "
            f"global_st_sectors={profile.global_mem.store_sectors:.0f} "
            f"smem_conflicts={profile.shared_mem.total_conflicts:.0f} "
            f"local_ld={profile.local_mem.load_sectors:.0f} "
            f"local_st={profile.local_mem.store_sectors:.0f}"
        )

    return profiles


def _build_profile(kernel_name: str, rows: list) -> NcuProfile:
    """Build one NcuProfile from one or more NcuKernelRow for the same kernel."""
    # Merge all metrics from all rows (last-write wins per metric). This is
    # appropriate for metric-per-row CSV; NCU repeats the same metric value in
    # a row only when it belongs to a distinct launch, which cannot be safely
    # summed without launch identity in the normalized schema.
    merged: dict[str, str] = {}
    units: dict[str, str] = {}
    for row in rows:
        merged.update(row.metrics)
        units.update(getattr(row, "metric_units", {}))

    profile = NcuProfile(kernel_name=kernel_name, raw_metrics=merged)
    missing = []

    def resolve(field_name: str) -> Optional[float]:
        """Try each candidate metric name and return the first found value."""
        candidates = METRIC_MAP.get(field_name, [])
        for candidate in candidates:
            if candidate in merged:
                val_str = merged[candidate]
                val = _parse_float(val_str)
                if val is not None:
                    if field_name == "duration_us":
                        return _duration_to_us(val, units.get(candidate, ""), candidate)
                    return val
        missing.append(field_name)
        return None

    # --- Global memory ---
    gm = profile.global_mem
    gm.load_sectors   = resolve("global_load_sectors")  or 0.0
    gm.store_sectors  = resolve("global_store_sectors") or 0.0
    gm.load_requests  = resolve("global_load_requests") or 0.0
    gm.store_requests = resolve("global_store_requests") or 0.0
    gm.compute_ratios()

    # --- Shared memory ---
    sm = profile.shared_mem
    sm.bank_conflicts_ld = resolve("smem_bank_conflict_ld") or 0.0
    sm.bank_conflicts_st = resolve("smem_bank_conflict_st") or 0.0

    # --- Local memory ---
    lm = profile.local_mem
    lm.load_sectors  = resolve("local_load_sectors")  or 0.0
    lm.store_sectors = resolve("local_store_sectors") or 0.0

    # --- Duration ---
    dur = resolve("duration_us")
    if dur is not None:
        profile.duration_us = dur

    profile.missing_metrics = missing

    if missing:
        logger.debug(f"[{kernel_name}] Missing metrics: {missing}")

    return profile


def _parse_float(val: str) -> Optional[float]:
    """Parse a metric value string to float, handling commas and units."""
    if not val or val.strip() == "":
        return None
    cleaned = val.replace(",", "").strip()
    if cleaned.lower() in {"n/a", "na", "not available", "--"}:
        return None
    m = re.match(r"[-+]?\d*\.?\d+", cleaned)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _duration_to_us(value: float, unit: str, metric_name: str) -> float:
    """Convert an NCU duration value to microseconds when its unit is known."""
    normalized = unit.strip().lower().replace("µ", "u")
    factors = {
        "second": 1_000_000.0, "seconds": 1_000_000.0, "s": 1_000_000.0,
        "msecond": 1_000.0, "milliseconds": 1_000.0, "ms": 1_000.0,
        "usecond": 1.0, "microseconds": 1.0, "us": 1.0,
        "nsecond": 0.001, "nanoseconds": 0.001, "ns": 0.001,
    }
    if normalized in factors:
        return value * factors[normalized]
    # gpu__time_duration is conventionally reported by NCU in nanoseconds.
    # Do not guess for display-only "Duration" values without a unit.
    if metric_name == "gpu__time_duration.sum":
        return value * 0.001
    return value


def _clean_kernel_name(name: str) -> str:
    """Strip quotes and template noise from kernel names."""
    name = name.strip().strip('"')
    # Remove template args for readability: foo<int, 4> -> foo
    # But keep the base name meaningful
    name = re.sub(r"<[^>]*>", "", name).strip()
    # NCU commonly reports demangled signatures, while ptxas gives only the
    # function identifier. Keeping the identifier makes resource correlation
    # deterministic without pretending overloads are distinguishable in v1.
    return re.sub(r"\(.*$", "", name).strip()
