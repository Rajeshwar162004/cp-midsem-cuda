"""
nvcc_runner.py
============================================================
Compiles a CUDA source file using nvcc and captures all output.

Responsibilities:
  - Run nvcc with appropriate flags (lineinfo, resource-usage, arch)
  - Capture stdout, stderr, return code
  - Parse ptxas resource-usage output from stderr
  - Return a structured CompileResult
"""

import subprocess
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────

@dataclass
class PtxasInfo:
    """Parsed ptxas resource-usage output for one kernel."""
    kernel_mangled: str = ""          # mangled C++ symbol
    kernel_display:  str = ""          # prettified name (best-effort)
    registers:       int = 0
    stack_frame_bytes:  int = 0        # local memory via stack (runtime-indexed arrays)
    spill_stores_bytes: int = 0
    spill_loads_bytes:  int = 0
    shared_mem_bytes:   int = 0
    cmem_bytes:         int = 0

    @property
    def has_local_memory(self) -> bool:
        return (self.stack_frame_bytes > 0 or
                self.spill_stores_bytes > 0 or
                self.spill_loads_bytes > 0)


@dataclass
class CompileResult:
    success:    bool
    source:     str          # input .cu path
    binary:     str          # output binary path
    command:    str          # full nvcc command used
    stdout:     str = ""
    stderr:     str = ""
    returncode: int = -1
    ptxas_info: list = field(default_factory=list)   # list[PtxasInfo]
    error_msg:  str = ""


# ── Ptxas parser ────────────────────────────────────────────

def _parse_ptxas_output(stderr: str) -> list[PtxasInfo]:
    """
    Parse ptxas resource-usage lines from nvcc stderr.

    Example ptxas output block:
        ptxas info    : Compiling entry function '_Z18bad_register_spillPKfPfi' for 'sm_89'
        ptxas info    : Function properties for _Z18bad_register_spillPKfPfi
            512 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
        ptxas info    : Used 56 registers, 512 bytes cumulative stack size, 372 bytes cmem[0]
    """
    results: list[PtxasInfo] = []
    lines = stderr.splitlines()

    # Pattern: entry function name
    re_entry = re.compile(
        r"ptxas info\s*:\s*Compiling entry function\s+'([^']+)'"
    )
    # Pattern: stack frame / spill
    re_frame = re.compile(
        r"(\d+)\s+bytes\s+stack\s+frame,\s*(\d+)\s+bytes\s+spill\s+stores,\s*(\d+)\s+bytes\s+spill\s+loads"
    )
    # Pattern: registers + cmem
    re_regs = re.compile(
        r"Used\s+(\d+)\s+registers"
    )
    # Pattern: shared memory (appears on the "Used" line sometimes)
    re_smem = re.compile(
        r"(\d+)\s+bytes\s+smem"
    )

    current: Optional[PtxasInfo] = None

    for line in lines:
        m_entry = re_entry.search(line)
        if m_entry:
            # Save previous entry if any
            if current:
                results.append(current)
            mangled = m_entry.group(1)
            current = PtxasInfo(
                kernel_mangled=mangled,
                kernel_display=_demangle_simple(mangled),
            )
            continue

        if current is None:
            continue

        m_frame = re_frame.search(line)
        if m_frame:
            current.stack_frame_bytes  = int(m_frame.group(1))
            current.spill_stores_bytes = int(m_frame.group(2))
            current.spill_loads_bytes  = int(m_frame.group(3))
            continue

        m_regs = re_regs.search(line)
        if m_regs:
            current.registers = int(m_regs.group(1))
            m_smem = re_smem.search(line)
            if m_smem:
                current.shared_mem_bytes = int(m_smem.group(1))
            continue

    if current:
        results.append(current)

    return results


def _demangle_simple(mangled: str) -> str:
    """
    Best-effort simple demangling without c++filt dependency.
    Strips leading _Z and numeric-prefixed segments.
    E.g. _Z18bad_register_spillPKfPfi -> bad_register_spill
    """
    import re
    # Try to extract the first identifier after _Z + number
    m = re.match(r'_Z\d*(\d+)([A-Za-z_]\w*)', mangled)
    if m:
        return m.group(2)
    return mangled


# ── Main runner ─────────────────────────────────────────────

def compile_cuda(
    source_path:    str,
    output_path:    Optional[str] = None,
    arch:           str = "sm_89",
    extra_flags:    list[str] = None,
    resource_usage: bool = True,
    lineinfo:       bool = True,
    optimize:       int = 2,
    timeout:        int = 120,
) -> CompileResult:
    """
    Compile a CUDA source file with nvcc.

    Args:
        source_path:    Path to the .cu file.
        output_path:    Path for the compiled binary. Defaults to source stem.
        arch:           GPU architecture, e.g. "sm_89".
        extra_flags:    Additional nvcc flags.
        resource_usage: Pass --resource-usage to get ptxas info.
        lineinfo:       Pass -lineinfo for source correlation in profiler.
        optimize:       Optimization level (0-3).
        timeout:        Max seconds to wait for compilation.

    Returns:
        CompileResult with all captured info.
    """
    src = Path(source_path).resolve()
    if not src.exists():
        return CompileResult(
            success=False, source=str(src), binary="", command="",
            error_msg=f"Source file not found: {src}"
        )

    if output_path is None:
        output_path = str(src.with_suffix(""))

    cmd = ["nvcc", f"-arch={arch}", f"-O{optimize}"]

    if lineinfo:
        cmd.append("-lineinfo")
    if resource_usage:
        cmd.append("--resource-usage")
    if extra_flags:
        cmd.extend(extra_flags)

    cmd += [str(src), "-o", str(output_path)]

    cmd_str = " ".join(cmd)
    logger.info(f"Compiling: {cmd_str}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return CompileResult(
            success=False, source=str(src), binary="", command=cmd_str,
            error_msg="nvcc not found. Is CUDA installed and nvcc in PATH?"
        )
    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False, source=str(src), binary="", command=cmd_str,
            error_msg=f"Compilation timed out after {timeout}s"
        )

    combined_stderr = proc.stderr or ""
    ptxas_list = []
    if resource_usage and combined_stderr:
        ptxas_list = _parse_ptxas_output(combined_stderr)

    success = (proc.returncode == 0)

    result = CompileResult(
        success=success,
        source=str(src),
        binary=str(output_path) if success else "",
        command=cmd_str,
        stdout=proc.stdout or "",
        stderr=combined_stderr,
        returncode=proc.returncode,
        ptxas_info=ptxas_list,
        error_msg="" if success else f"nvcc exited with code {proc.returncode}",
    )

    if success:
        logger.info(f"Compilation succeeded → {output_path}")
        for p in ptxas_list:
            logger.debug(
                f"  [{p.kernel_display}] regs={p.registers} "
                f"stack={p.stack_frame_bytes}B "
                f"spill_ld={p.spill_loads_bytes}B "
                f"spill_st={p.spill_stores_bytes}B"
            )
    else:
        logger.error(f"Compilation failed (rc={proc.returncode}):\n{combined_stderr}")

    return result
