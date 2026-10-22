"""
ncu_runner.py
============================================================
Runs NVIDIA Nsight Compute CLI (ncu) on a compiled CUDA binary
and captures its output in CSV and raw text formats.

Responsibilities:
  - Detect ncu and verify it is callable
  - Run ncu with a configurable set of metric sections
  - Capture CSV output (--csv flag) for structured parsing
  - Store the .ncu-rep binary report file
  - Return structured NcuResult
"""

import subprocess
import logging
import csv
import io
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────

@dataclass
class NcuKernelRow:
    """One row from ncu CSV output, representing one kernel invocation."""
    kernel_name: str = ""
    # Raw metrics dict: metric_name -> value string
    metrics: dict = field(default_factory=dict)
    metric_units: dict = field(default_factory=dict)

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Safely get a metric as float, stripping commas/units."""
        val = self.metrics.get(key, "")
        if val == "" or val is None:
            return default
        # Remove commas (thousand separators) and strip trailing units
        cleaned = str(val).replace(",", "").strip()
        # Take the numeric prefix if there are trailing non-numeric chars
        import re
        m = re.match(r"[-+]?\d*\.?\d+", cleaned)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return default
        return default


@dataclass
class NcuResult:
    success:    bool
    binary:     str
    command:    str
    stdout:     str = ""
    stderr:     str = ""
    returncode: int = -1
    csv_text:   str = ""
    report_file: str = ""
    kernel_rows: list = field(default_factory=list)   # list[NcuKernelRow]
    error_msg:  str = ""


# ── Default metric sections ──────────────────────────────────
# These sections are available on sm_89 (Ada Lovelace) with NCU 2024.1.
# We query sections rather than individual metrics to be architecture-agnostic.
# The parser will then extract what's available.

DEFAULT_SECTIONS = [
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Chart",
    "SpeedOfLight",
    "SpeedOfLight_RooflineChart",
    "WarpStateStats",      # NCU 2024.1: identifier is WarpStateStats, not WarpStateStatistics
    "Occupancy",
    "SourceCounters",
    "LaunchStats",         # NCU 2024.1: identifier is LaunchStats, not LaunchStatistics
]

# Fallback: if sections fail, use a minimal explicit metric list
FALLBACK_METRICS = [
    # Global memory
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    # Local memory (spill traffic)
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    # Shared memory bank conflicts
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Duration
    "sm__cycles_elapsed.avg",
    "gpu__time_duration.sum",
]


# ── Runner ───────────────────────────────────────────────────

def run_ncu(
    binary_path:    str,
    binary_args:    list[str] = None,
    output_dir:     Optional[str] = None,
    report_name:    str = "ncu_report",
    sections:       list[str] = None,
    kernel_filter:  Optional[str] = None,
    use_sections:   bool = True,
    timeout:        int = 300,
) -> NcuResult:
    """
    Run Nsight Compute on a compiled CUDA binary.

    Args:
        binary_path:   Path to the compiled CUDA executable.
        binary_args:   Arguments to pass to the binary.
        output_dir:    Directory to store ncu output files.
        report_name:   Base name for .ncu-rep and CSV files.
        sections:      NCU sections to collect. Defaults to DEFAULT_SECTIONS.
        kernel_filter: Optional kernel name substring to filter (--kernel-name).
        use_sections:  If True, use --section flags. If False, use --metrics fallback.
        timeout:       Max seconds for profiling run.

    Returns:
        NcuResult with parsed kernel rows and raw output.
    """
    binary = Path(binary_path).resolve()
    if not binary.exists():
        return NcuResult(
            success=False, binary=str(binary), command="",
            error_msg=f"Binary not found: {binary}"
        )

    if not os.access(str(binary), os.X_OK):
        return NcuResult(
            success=False, binary=str(binary), command="",
            error_msg=f"Binary not executable: {binary}"
        )

    if output_dir is None:
        output_dir = str(binary.parent / "ncu_output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_path = str(Path(output_dir) / f"{report_name}.ncu-rep")
    csv_path    = str(Path(output_dir) / f"{report_name}.csv")

    if sections is None:
        sections = DEFAULT_SECTIONS

    # Build ncu command
    cmd = [
        "ncu",
        "--export", report_path,
        "--force-overwrite",
        "--csv",
    ]

    if use_sections:
        for s in sections:
            cmd += ["--section", s]
    else:
        # Fallback: explicit metrics
        cmd += ["--metrics", ",".join(FALLBACK_METRICS)]

    if kernel_filter:
        cmd += ["--kernel-name", kernel_filter]

    cmd.append(str(binary))
    if binary_args:
        cmd.extend(binary_args)

    cmd_str = " ".join(cmd)
    logger.info(f"Profiling: {cmd_str}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return NcuResult(
            success=False, binary=str(binary), command=cmd_str,
            error_msg="ncu not found. Install Nsight Compute and ensure it is in PATH."
        )
    except subprocess.TimeoutExpired:
        return NcuResult(
            success=False, binary=str(binary), command=cmd_str,
            error_msg=f"ncu timed out after {timeout}s"
        )

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # The profiling invocation writes its measurements into the .ncu-rep
    # report. With NCU 2024.1 it does not also emit a CSV table to stdout, so
    # export the completed report in a second, read-only invocation.
    report_exists = Path(report_path).exists()
    export_returncode = 0
    csv_export_error = ""
    if report_exists:
        export_cmd = ["ncu", "--import", report_path, "--csv", "--page", "raw"]
        try:
            export_proc = subprocess.run(
                export_cmd, capture_output=True, text=True, timeout=timeout
            )
            export_returncode = export_proc.returncode
            csv_text = _extract_csv_payload(export_proc.stdout or "")
            csv_export_error = export_proc.stderr or ""
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            export_returncode = -1
            csv_text = ""
            csv_export_error = f"Could not export NCU report as CSV: {error}"
    else:
        # Retain compatibility with NCU versions that print CSV during the
        # profiling invocation itself.
        csv_text = _extract_csv_payload(stdout)
    if csv_text:
        try:
            with open(csv_path, "w") as f:
                f.write(csv_text)
            logger.info(f"CSV saved → {csv_path}")
        except OSError as e:
            logger.warning(f"Could not write CSV: {e}")

    # Parse kernel rows from CSV
    kernel_rows = _parse_ncu_csv(csv_text) if csv_text else []

    # ncu returns non-zero on some profiling issues even when it succeeds,
    # so we consider it successful if we got at least some output
    success = ((proc.returncode == 0 and export_returncode == 0) or len(kernel_rows) > 0)

    if not success and not stderr and not stdout:
        error_msg = f"ncu exited with code {proc.returncode} and no output"
    elif proc.returncode != 0 and len(kernel_rows) == 0:
        # Check for common permission issue
        combined_output = f"{stdout}\n{stderr}"
        if "ERR_NVGPUCTRPERM" in combined_output:
            error_msg = (
                "ERR_NVGPUCTRPERM: this user cannot access NVIDIA GPU performance "
                "counters. Run the profiling command with sudo, or ask an administrator "
                "to enable non-admin counter access."
            )
        else:
            error_msg = f"ncu exited with code {proc.returncode}"
    elif export_returncode != 0:
        error_msg = f"ncu could not export CSV from its report: {csv_export_error.strip()}"
    elif not kernel_rows:
        error_msg = "ncu created a report but no kernel rows could be exported as CSV"
    else:
        error_msg = ""

    result = NcuResult(
        success=success,
        binary=str(binary),
        command=cmd_str,
        stdout=stdout,
        stderr=stderr,
        returncode=proc.returncode,
        csv_text=csv_text,
        report_file=report_path if Path(report_path).exists() else "",
        kernel_rows=kernel_rows,
        error_msg=error_msg,
    )

    if kernel_rows:
        logger.info(f"Parsed {len(kernel_rows)} kernel invocation(s) from ncu output")
    else:
        logger.warning("No kernel rows parsed from ncu output")

    return result


def _extract_csv_payload(stdout: str) -> str:
    """Return only quoted CSV records from mixed NCU stdout.

    ``ncu --csv`` still emits ``==PROF==`` status messages and the target
    program's normal stdout. NCU's CSV records are quoted, so retaining only
    quoted lines avoids treating those messages as a CSV header.
    """
    return "\n".join(line for line in stdout.splitlines() if line.lstrip().startswith('"'))


# ── CSV parser ───────────────────────────────────────────────

def _parse_ncu_csv(csv_text: str) -> list[NcuKernelRow]:
    """
    Parse Nsight Compute CSV output into NcuKernelRow objects.

    NCU CSV format (--csv flag) has two header rows:
      Row 1: "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",...,"Metric Name"
      Row 2: "","","","","","","","","","","","","Metric Unit"
      Row 3+: data rows

    Actually, ncu --csv outputs in a different format depending on version.
    We handle both the flat format and the metric-per-row format.
    """
    if not csv_text.strip():
        return []

    rows: list[NcuKernelRow] = []

    try:
        reader = csv.reader(io.StringIO(csv_text))
        all_rows = list(reader)
    except Exception as e:
        logger.warning(f"CSV parse error: {e}")
        return []

    if len(all_rows) < 2:
        return []

    # Detect format by inspecting headers
    headers = [h.strip().strip('"') for h in all_rows[0]]

    # NCU CSV format: columns include "Kernel Name", "Metric Name", "Metric Value".
    if "Metric Name" in headers and "Metric Value" in headers:
        rows = _parse_metric_per_row_format(all_rows, headers)
    elif "Kernel Name" in headers:
        rows = _parse_wide_format(all_rows, headers)
    else:
        logger.warning(f"Unknown NCU CSV format. Headers: {headers[:10]}")

    return rows


def _parse_metric_per_row_format(
    all_rows: list[list[str]], headers: list[str]
) -> list[NcuKernelRow]:
    """
    Format where each row is: KernelName, ..., MetricName, MetricUnit, MetricValue
    """
    rows: list[NcuKernelRow] = []
    kernel_col = headers.index("Kernel Name") if "Kernel Name" in headers else -1
    metric_name_col  = headers.index("Metric Name")  if "Metric Name"  in headers else -1
    metric_value_col = headers.index("Metric Value") if "Metric Value" in headers else -1
    metric_unit_col = headers.index("Metric Unit") if "Metric Unit" in headers else -1

    if kernel_col < 0 or metric_name_col < 0 or metric_value_col < 0:
        return []

    # Group by kernel name (they repeat for each metric)
    kernel_map: dict[str, NcuKernelRow] = {}

    for row in all_rows[1:]:  # skip header
        if len(row) <= max(kernel_col, metric_name_col, metric_value_col):
            continue
        kernel = row[kernel_col].strip().strip('"')
        metric = row[metric_name_col].strip().strip('"')
        value  = row[metric_value_col].strip().strip('"')

        if not kernel:
            continue

        if kernel not in kernel_map:
            kernel_map[kernel] = NcuKernelRow(kernel_name=kernel)
        kernel_map[kernel].metrics[metric] = value
        if metric_unit_col >= 0 and len(row) > metric_unit_col:
            kernel_map[kernel].metric_units[metric] = row[metric_unit_col].strip().strip('"')

    return list(kernel_map.values())


def _parse_wide_format(
    all_rows: list[list[str]], headers: list[str]
) -> list[NcuKernelRow]:
    """
    Format where each row is one kernel launch and columns are metric names.
    """
    rows: list[NcuKernelRow] = []
    kernel_col = headers.index("Kernel Name") if "Kernel Name" in headers else 0

    # Skip unit row if present (second row with unit labels)
    data_start = 1
    metric_units = {}
    if len(all_rows) > 1:
        second_row = [c.strip().strip('"') for c in all_rows[1]]
        # If second row looks like units (ns, bytes, etc.), skip it
        if all(not c.replace(".", "").replace("-", "").isdigit() for c in second_row[1:5]):
            data_start = 2
            metric_units = {
                headers[i]: second_row[i]
                for i in range(min(len(headers), len(second_row)))
            }

    for row in all_rows[data_start:]:
        if len(row) < len(headers):
            continue
        cells = [c.strip().strip('"') for c in row]
        kernel_name = cells[kernel_col] if kernel_col < len(cells) else "unknown"
        metrics = {headers[i]: cells[i] for i in range(len(headers)) if i < len(cells)}
        units = dict(metric_units)
        for i, header in enumerate(headers):
            if header.endswith(" Unit") and i < len(cells):
                units[header[:-5]] = cells[i]
        rows.append(NcuKernelRow(kernel_name=kernel_name, metrics=metrics, metric_units=units))

    return rows
