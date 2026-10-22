"""
profile.py
============================================================
Week 2 entry point: fully automated CUDA profiling pipeline.

Usage:
    python profiler/profile.py <source.cu> [options]

Pipeline:
    1. Environment check (nvidia-smi, nvcc, ncu)
    2. Compile with nvcc (--resource-usage, -lineinfo)
    3. Run the compiled binary
    4. Run Nsight Compute (ncu --csv)
    5. Parse ptxas output (register/spill info)
    6. Parse ncu CSV output (memory metrics)
    7. Assemble normalized profile_result.json
    8. Save all raw artifacts to results/<kernel_name>/

Example:
    python profiler/profile.py benchmarks/register_spill/bad_spill.cu
    python profiler/profile.py benchmarks/global_access/bad_global.cu --arch sm_89
    python profiler/profile.py benchmarks/bank_conflict/bad_bank.cu --output results/mytest
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Resolve paths relative to this file so the script works from any cwd
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from profiler.nvcc_runner import compile_cuda, CompileResult
from profiler.ncu_runner  import run_ncu,      NcuResult
from parser.ncu_parser    import parse_ncu_rows
from parser.ptxas_parser  import parse_ptxas_output, PtxasReport


# ── Logging setup ────────────────────────────────────────────

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


# ── Environment check ────────────────────────────────────────

def check_environment() -> dict:
    """
    Verify that required tools are available and return system info.
    Exits with a helpful message if critical tools are missing.
    """
    info = {}
    errors = []

    tools = {
        "nvidia-smi": ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                        "--format=csv,noheader"],
        "nvcc":        ["nvcc", "--version"],
        "ncu":         ["ncu",  "--version"],
    }

    optional = {"nsys": ["nsys", "--version"]}

    log = logging.getLogger("env_check")

    for name, cmd in tools.items():
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                output = (r.stdout or r.stderr or "").strip().splitlines()
                info[name] = output[0] if output else "ok"
                log.info(f"  ✓ {name}: {info[name]}")
            else:
                errors.append(f"{name} returned non-zero exit code")
        except FileNotFoundError:
            errors.append(f"{name} not found in PATH")
        except subprocess.TimeoutExpired:
            errors.append(f"{name} timed out")

    for name, cmd in optional.items():
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            info[name] = (r.stdout or r.stderr or "").strip().splitlines()[0]
            log.info(f"  ✓ {name} (optional): {info[name]}")
        except (FileNotFoundError, subprocess.TimeoutExpired, IndexError):
            info[name] = "not available"
            log.info(f"  ○ {name}: not available (optional)")

    if errors:
        print("\n[ERROR] Missing required tools:")
        for e in errors:
            print(f"  - {e}")
        print("\nEnsure CUDA toolkit and Nsight Compute are installed and in PATH.")
        sys.exit(1)

    # Parse GPU info from nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,compute_cap,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            info["gpu_name"]      = parts[0].strip() if len(parts) > 0 else "unknown"
            info["driver"]        = parts[1].strip() if len(parts) > 1 else "unknown"
            info["compute_cap"]   = parts[2].strip() if len(parts) > 2 else "unknown"
            info["gpu_memory_mb"] = parts[3].strip() if len(parts) > 3 else "unknown"
    except Exception:
        pass

    return info


# ── Raw binary runner ────────────────────────────────────────

def run_binary(binary_path: str, timeout: int = 60) -> dict:
    """Run the compiled binary and capture its output."""
    log = logging.getLogger("run_binary")
    try:
        t_start = time.perf_counter()
        r = subprocess.run(
            [binary_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t_start
        log.info(f"Binary exited with code {r.returncode} in {elapsed:.2f}s")
        return {
            "success":    r.returncode == 0,
            "returncode": r.returncode,
            "stdout":     r.stdout or "",
            "stderr":     r.stderr or "",
            "elapsed_s":  round(elapsed, 3),
        }
    except FileNotFoundError:
        return {"success": False, "error": f"Binary not found: {binary_path}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Binary timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Profile result assembler ─────────────────────────────────

def assemble_profile_result(
    system_info:    dict,
    source_path:    str,
    binary_path:    str,
    compile_result: CompileResult,
    run_result:     dict,
    ncu_result:     NcuResult,
    ptxas_report:   PtxasReport,
    ncu_profiles:   list,
) -> dict:
    """
    Assemble a normalized profile_result.json from all pipeline outputs.

    This is the canonical data model that detectors (Week 3) will consume.
    """
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Build per-kernel data
    kernels_data = []
    for profile in ncu_profiles:
        # Find matching ptxas info
        ptxas_k = ptxas_report.get_kernel(profile.kernel_name)

        kernel_entry = {
            "name":        profile.kernel_name,
            "duration_us": profile.duration_us,

            "resources": {
                "registers_per_thread": ptxas_k.registers          if ptxas_k else None,
                "stack_frame_bytes":    ptxas_k.stack_frame_bytes   if ptxas_k else None,
                "spill_stores_bytes":   ptxas_k.spill_stores_bytes  if ptxas_k else None,
                "spill_loads_bytes":    ptxas_k.spill_loads_bytes   if ptxas_k else None,
                "shared_mem_bytes":     ptxas_k.shared_mem_bytes    if ptxas_k else None,
            },

            "global_memory": {
                "load_sectors":       profile.global_mem.load_sectors,
                "store_sectors":      profile.global_mem.store_sectors,
                "load_requests":      profile.global_mem.load_requests,
                "store_requests":     profile.global_mem.store_requests,
                "load_sector_ratio":  round(profile.global_mem.load_sector_ratio,  2),
                "store_sector_ratio": round(profile.global_mem.store_sector_ratio, 2),
                # Ratios > 1 suggest uncoalesced access (ideal = 1.0)
                "efficiency_note": (
                    "POSSIBLY UNCOALESCED"
                    if profile.global_mem.load_sector_ratio > 2 or
                       profile.global_mem.store_sector_ratio > 2
                    else "ok"
                ),
            },

            "shared_memory": {
                "bank_conflicts_ld":  profile.shared_mem.bank_conflicts_ld,
                "bank_conflicts_st":  profile.shared_mem.bank_conflicts_st,
                "total_conflicts":    profile.shared_mem.total_conflicts,
            },

            "local_memory": {
                "load_sectors":  profile.local_mem.load_sectors,
                "store_sectors": profile.local_mem.store_sectors,
                "has_traffic":   profile.local_mem.has_traffic,
            },

            "missing_ncu_metrics": profile.missing_metrics,
        }

        # If we have ptxas info, add local memory flag
        if ptxas_k:
            kernel_entry["resources"]["has_local_memory"] = ptxas_k.has_local_memory

        kernels_data.append(kernel_entry)

    # Also include ptxas-only kernels (kernels that ran but weren't profiled by ncu)
    profiled_names = {p.kernel_name.split("(", 1)[0].strip() for p in ncu_profiles}
    for k in ptxas_report.kernels:
        if k.kernel_display not in profiled_names and k.kernel_mangled not in profiled_names:
            kernels_data.append({
                "name": k.kernel_display,
                "source": "ptxas_only",
                "resources": {
                    "registers_per_thread": k.registers,
                    "stack_frame_bytes":    k.stack_frame_bytes,
                    "spill_stores_bytes":   k.spill_stores_bytes,
                    "spill_loads_bytes":    k.spill_loads_bytes,
                    "shared_mem_bytes":     k.shared_mem_bytes,
                    "has_local_memory":     k.has_local_memory,
                },
            })

    return {
        "schema_version": "1.0",
        "timestamp":      now,

        "system": {
            "gpu":          system_info.get("gpu_name",    "unknown"),
            "driver":       system_info.get("driver",      "unknown"),
            "compute_cap":  system_info.get("compute_cap", "unknown"),
            "gpu_memory":   system_info.get("gpu_memory_mb", "unknown"),
            "nvcc":         system_info.get("nvcc",  "unknown"),
            "ncu":          system_info.get("ncu",   "unknown"),
            "nsys":         system_info.get("nsys",  "not available"),
        },

        "application": {
            "source":       source_path,
            "binary":       binary_path,
            "compile_cmd":  compile_result.command,
            "run_success":  run_result.get("success", False),
            "run_elapsed_s": run_result.get("elapsed_s", None),
            "run_returncode": run_result.get("returncode", None),
        },

        "kernels": kernels_data,

        "raw_artifacts": {
            "ncu_report":    ncu_result.report_file,
            "ncu_success":   ncu_result.success,
            "ncu_error":     ncu_result.error_msg,
        },
    }


# ── Output helpers ───────────────────────────────────────────

def save_results(output_dir: str, profile_result: dict,
                 compile_result: CompileResult,
                 ncu_result: NcuResult) -> None:
    """Save all artifacts to the output directory."""
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)

    # Normalized JSON
    json_path = d / "profile_result.json"
    with open(json_path, "w") as f:
        json.dump(profile_result, f, indent=2)
    logging.getLogger("save").info(f"Saved profile_result.json → {json_path}")

    # Raw compiler output
    with open(d / "compiler_output.txt", "w") as f:
        f.write(f"COMMAND:\n{compile_result.command}\n\n")
        f.write(f"STDOUT:\n{compile_result.stdout}\n\n")
        f.write(f"STDERR:\n{compile_result.stderr}\n")

    # Raw ncu CSV
    if ncu_result.csv_text:
        with open(d / "ncu_output.csv", "w") as f:
            f.write(ncu_result.csv_text)

    # Preserve NCU progress messages and the profiled application's stdout.
    if ncu_result.stdout or ncu_result.stderr:
        with open(d / "ncu_output.txt", "w") as f:
            f.write("STDOUT:\n" + ncu_result.stdout)
            f.write("\n\nSTDERR:\n" + ncu_result.stderr)

    # Metadata
    meta = {
        "timestamp":    profile_result["timestamp"],
        "source":       profile_result["application"]["source"],
        "gpu":          profile_result["system"]["gpu"],
        "ncu_report":   ncu_result.report_file,
        "output_dir":   str(d.resolve()),
    }
    with open(d / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def print_summary(profile_result: dict) -> None:
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 62)
    print("  GPU MEMORY TOOLKIT — PROFILE SUMMARY")
    print("=" * 62)

    sys_info = profile_result.get("system", {})
    print(f"\nGPU:    {sys_info.get('gpu', '?')}")
    print(f"Driver: {sys_info.get('driver', '?')}   "
          f"Compute: {sys_info.get('compute_cap', '?')}")

    app = profile_result.get("application", {})
    print(f"Source: {app.get('source', '?')}")

    kernels = profile_result.get("kernels", [])
    if not kernels:
        print("\n[!] No kernel data collected.")
        return

    for k in kernels:
        print(f"\n{'─' * 62}")
        print(f"  Kernel: {k['name']}")
        dur = k.get("duration_us", 0)
        if dur:
            print(f"  Duration: {dur:.2f} µs")

        res = k.get("resources", {})
        if res.get("registers_per_thread") is not None:
            print(f"  Registers/thread: {res['registers_per_thread']}")
        if res.get("has_local_memory"):
            print(f"  ⚠ LOCAL MEMORY: {res.get('stack_frame_bytes', 0)}B stack + "
                  f"{res.get('spill_stores_bytes', 0)}B spill_st + "
                  f"{res.get('spill_loads_bytes', 0)}B spill_ld")

        gm = k.get("global_memory", {})
        if gm.get("load_sector_ratio", 0) > 0:
            print(f"  Global load  sector ratio: {gm['load_sector_ratio']:.2f}  "
                  f"(ideal=1.0, >2 = suspect)")
        if gm.get("store_sector_ratio", 0) > 0:
            print(f"  Global store sector ratio: {gm['store_sector_ratio']:.2f}  "
                  f"(ideal=1.0, >2 = suspect)")
        if gm.get("efficiency_note") == "POSSIBLY UNCOALESCED":
            print(f"  ⚠ POSSIBLY UNCOALESCED GLOBAL ACCESS")

        sm = k.get("shared_memory", {})
        if sm.get("total_conflicts", 0) > 0:
            print(f"  ⚠ SHARED MEMORY BANK CONFLICTS: {sm['total_conflicts']:.0f}")

        lm = k.get("local_memory", {})
        if lm.get("has_traffic"):
            print(f"  ⚠ LOCAL MEMORY TRAFFIC: "
                  f"ld_sectors={lm.get('load_sectors', 0):.0f} "
                  f"st_sectors={lm.get('store_sectors', 0):.0f}")

    print("\n" + "=" * 62)


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU Memory Profiling Toolkit — automated profiling pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", help="Path to the .cu CUDA source file")
    parser.add_argument(
        "--arch", default=None,
        help="CUDA architecture (for example sm_86). Defaults to the detected GPU capability.",
    )
    parser.add_argument("--output",  default=None,    help="Output directory for results")
    parser.add_argument("--no-ncu",  action="store_true", help="Skip Nsight Compute profiling")
    parser.add_argument("--no-run",  action="store_true", help="Skip running the binary")
    parser.add_argument("--kernel",  default=None, help="NCU kernel name filter")
    parser.add_argument("--maxreg",  default=None, type=int, help="--maxrregcount for nvcc")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--json-only", action="store_true", help="Only print JSON, no summary")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    log = logging.getLogger("profile")

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"[ERROR] Source file not found: {source}")
        sys.exit(1)

    # Output dir defaults to results/<source_stem>
    if args.output:
        output_dir = args.output
    else:
        results_root = _ROOT / "results"
        output_dir   = str(results_root / source.stem)

    print(f"\n[1/6] Checking environment...")
    system_info = check_environment()

    arch = args.arch
    if arch is None:
        capability = str(system_info.get("compute_cap", "")).strip()
        if capability and capability != "unknown":
            arch = "sm_" + capability.replace(".", "")
        else:
            print("[ERROR] Could not detect compute capability; pass --arch (for example sm_86).")
            sys.exit(1)

    print(f"\n[2/6] Compiling {source.name}...")
    extra_flags = []
    if args.maxreg:
        extra_flags += [f"--maxrregcount={args.maxreg}"]

    binary_path = str(Path(output_dir) / source.stem)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    compile_result = compile_cuda(
        source_path=str(source),
        output_path=binary_path,
        arch=arch,
        extra_flags=extra_flags,
        resource_usage=True,
        lineinfo=True,
    )

    if not compile_result.success:
        print(f"\n[ERROR] Compilation failed:\n{compile_result.stderr}")
        sys.exit(1)
    print(f"  ✓ Compiled → {compile_result.binary}")

    # Parse ptxas immediately from compile output
    ptxas_report = parse_ptxas_output(compile_result.stderr)
    if ptxas_report.kernels:
        print(f"  ptxas: found {len(ptxas_report.kernels)} kernel(s)")
        for k in ptxas_report.kernels:
            flag = "⚠ LOCAL MEM" if k.has_local_memory else "ok"
            print(f"    [{k.kernel_display}] {k.summary()}  {flag}")

    # Run binary
    run_result = {}
    if not args.no_run:
        print(f"\n[3/6] Running {source.stem}...")
        run_result = run_binary(binary_path)
        if run_result.get("success"):
            print(f"  ✓ Ran in {run_result.get('elapsed_s', '?')}s")
            if run_result.get("stdout"):
                # Print first few lines
                lines = run_result["stdout"].splitlines()[:6]
                for line in lines:
                    print(f"    {line}")
        else:
            print(f"  [!] Binary run failed: {run_result.get('error', run_result.get('stderr', ''))}")
    else:
        print(f"\n[3/6] Skipping binary run (--no-run)")
        run_result = {"success": None}

    # Run NCU
    ncu_result = NcuResult(success=False, binary=binary_path, command="",
                            error_msg="ncu skipped")
    ncu_profiles = []

    if not args.no_ncu:
        print(f"\n[4/6] Running Nsight Compute (ncu)...")
        ncu_result = run_ncu(
            binary_path=binary_path,
            output_dir=output_dir,
            report_name=source.stem,
            kernel_filter=args.kernel,
        )

        if ncu_result.success and ncu_result.kernel_rows:
            print(f"  ✓ ncu collected {len(ncu_result.kernel_rows)} kernel row(s)")
            print(f"\n[5/6] Parsing NCU metrics...")
            ncu_profiles = parse_ncu_rows(ncu_result.kernel_rows)
            print(f"  ✓ Normalized {len(ncu_profiles)} kernel profile(s)")
        else:
            print(f"  [!] ncu issue: {ncu_result.error_msg}")
            if ncu_result.stderr:
                for line in ncu_result.stderr.splitlines()[:5]:
                    print(f"      {line}")
            print("  Continuing with ptxas-only data...")
    else:
        print(f"\n[4/6] Skipping ncu (--no-ncu)")
        print(f"\n[5/6] Skipping ncu parse (--no-ncu)")

    print(f"\n[6/6] Assembling profile_result.json...")
    profile_result = assemble_profile_result(
        system_info=system_info,
        source_path=str(source),
        binary_path=binary_path,
        compile_result=compile_result,
        run_result=run_result,
        ncu_result=ncu_result,
        ptxas_report=ptxas_report,
        ncu_profiles=ncu_profiles,
    )

    save_results(output_dir, profile_result, compile_result, ncu_result)
    print(f"  ✓ Results saved to: {output_dir}")

    if args.json_only:
        print(json.dumps(profile_result, indent=2))
    else:
        print_summary(profile_result)
        print(f"\nFull JSON: {output_dir}/profile_result.json\n")


if __name__ == "__main__":
    main()
