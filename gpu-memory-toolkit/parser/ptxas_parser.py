"""
ptxas_parser.py
============================================================
Parses ptxas resource-usage output from nvcc stderr into a
structured PtxasReport.

This is the primary source for detecting register spilling.

ptxas output format (with --resource-usage):

  ptxas info    : Compiling entry function '_Z18bad_register_spillPKfPfi' for 'sm_89'
  ptxas info    : Function properties for _Z18bad_register_spillPKfPfi
      512 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 56 registers, 512 bytes cumulative stack size, 372 bytes cmem[0]

The "stack frame" is local memory used by runtime-indexed local arrays.
"spill stores/loads" is explicit register spilling from register pressure.
Both indicate local memory traffic (off-chip DRAM).
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data model ──────────────────────────────────────────────

@dataclass
class KernelResourceInfo:
    """Resource usage for one CUDA kernel as reported by ptxas."""
    kernel_mangled:    str = ""     # raw mangled symbol
    kernel_display:    str = ""     # best-effort demangled name

    # Register info
    registers:         int = 0

    # Local memory info
    # "stack frame" = local arrays accessed with runtime indices (guaranteed local mem)
    stack_frame_bytes: int = 0
    # "spill stores/loads" = register spilling from register pressure
    spill_stores_bytes: int = 0
    spill_loads_bytes:  int = 0

    # Shared memory info (bytes per block)
    shared_mem_bytes:   int = 0

    # Constant memory
    cmem_bytes:         int = 0

    @property
    def total_local_memory_bytes(self) -> int:
        """Total local memory = stack frame + spill."""
        return self.stack_frame_bytes + self.spill_stores_bytes + self.spill_loads_bytes

    @property
    def has_local_memory(self) -> bool:
        return self.total_local_memory_bytes > 0

    @property
    def has_spilling(self) -> bool:
        return self.spill_stores_bytes > 0 or self.spill_loads_bytes > 0

    @property
    def has_stack_frame(self) -> bool:
        return self.stack_frame_bytes > 0

    def summary(self) -> str:
        parts = [f"regs={self.registers}"]
        if self.stack_frame_bytes:
            parts.append(f"stack={self.stack_frame_bytes}B")
        if self.spill_stores_bytes:
            parts.append(f"spill_st={self.spill_stores_bytes}B")
        if self.spill_loads_bytes:
            parts.append(f"spill_ld={self.spill_loads_bytes}B")
        if self.shared_mem_bytes:
            parts.append(f"smem={self.shared_mem_bytes}B")
        return " | ".join(parts)


@dataclass
class PtxasReport:
    """Full ptxas parse result — may contain multiple kernels."""
    kernels: list = field(default_factory=list)   # list[KernelResourceInfo]
    raw_text: str = ""

    def get_kernel(self, name_fragment: str) -> Optional[KernelResourceInfo]:
        """Find a kernel by partial name match (case-insensitive)."""
        name_fragment = _kernel_identifier(name_fragment).lower()
        for k in self.kernels:
            if name_fragment == _kernel_identifier(k.kernel_display).lower() or \
               name_fragment in k.kernel_mangled.lower():
                return k
        return None

    def kernels_with_local_memory(self) -> list:
        return [k for k in self.kernels if k.has_local_memory]

    def kernels_with_spilling(self) -> list:
        return [k for k in self.kernels if k.has_spilling]


# ── Regex patterns ───────────────────────────────────────────

# ptxas info    : Compiling entry function '_Z4nameXYZ' for 'sm_89'
RE_ENTRY = re.compile(
    r"ptxas\s+info\s*:\s*Compiling\s+entry\s+function\s+'([^']+)'"
)

# "512 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
RE_FRAME = re.compile(
    r"(\d+)\s+bytes\s+stack\s+frame"
    r"(?:,\s*(\d+)\s+bytes\s+spill\s+stores)?"
    r"(?:,\s*(\d+)\s+bytes\s+spill\s+loads)?"
)

# "Used 56 registers, 512 bytes cumulative stack size, 372 bytes cmem[0]"
RE_REGS = re.compile(
    r"Used\s+(\d+)\s+registers"
    r"(?:,\s*(\d+)\s+bytes\s+cumulative\s+stack\s+size)?"
)

# "... 8192 bytes smem ..."
RE_SMEM = re.compile(r"(\d+)\s+bytes\s+smem")

# "... 372 bytes cmem[0] ..."
RE_CMEM = re.compile(r"(\d+)\s+bytes\s+cmem\[0\]")


# ── Parser ───────────────────────────────────────────────────

def parse_ptxas_output(stderr_text: str) -> PtxasReport:
    """
    Parse the full ptxas resource-usage output from nvcc stderr.

    Args:
        stderr_text: The complete stderr string from an nvcc --resource-usage run.

    Returns:
        PtxasReport containing one KernelResourceInfo per compiled kernel.
    """
    report = PtxasReport(raw_text=stderr_text)
    lines  = stderr_text.splitlines()

    current: Optional[KernelResourceInfo] = None

    for line in lines:
        # New kernel entry
        m_entry = RE_ENTRY.search(line)
        if m_entry:
            if current is not None:
                report.kernels.append(current)
            mangled = m_entry.group(1)
            current = KernelResourceInfo(
                kernel_mangled=mangled,
                kernel_display=_demangle(mangled),
            )
            continue

        if current is None:
            continue

        # Stack frame / spill line
        m_frame = RE_FRAME.search(line)
        if m_frame:
            current.stack_frame_bytes  = int(m_frame.group(1))
            current.spill_stores_bytes = int(m_frame.group(2) or 0)
            current.spill_loads_bytes  = int(m_frame.group(3) or 0)
            continue

        # Registers line
        m_regs = RE_REGS.search(line)
        if m_regs:
            current.registers = int(m_regs.group(1))

            # Shared memory
            m_smem = RE_SMEM.search(line)
            if m_smem:
                current.shared_mem_bytes = int(m_smem.group(1))

            # Constant memory
            m_cmem = RE_CMEM.search(line)
            if m_cmem:
                current.cmem_bytes = int(m_cmem.group(1))

            continue

    # Don't forget the last kernel
    if current is not None:
        report.kernels.append(current)

    # Log summary
    for k in report.kernels:
        flag = "⚠ LOCAL MEM" if k.has_local_memory else "✓ clean"
        logger.info(f"  [{k.kernel_display}] {k.summary()}  {flag}")

    return report


def parse_ptxas_from_compile_result(compile_result) -> PtxasReport:
    """
    Convenience wrapper: parse ptxas from a CompileResult object
    (as returned by nvcc_runner.compile_cuda).
    """
    stderr = getattr(compile_result, "stderr", "") or ""
    return parse_ptxas_output(stderr)


# ── Demangler ────────────────────────────────────────────────

def _demangle(mangled: str) -> str:
    """
    Best-effort demangling without c++filt.
    Handles the common Itanium ABI pattern: _Z<len><name><args>

    Examples:
      _Z18bad_register_spillPKfPfi  -> bad_register_spill
      _Z13good_no_spillPKfPfi        -> good_no_spill
      _Z16bad_bank_conflictPKfPfi    -> bad_bank_conflict
    """
    # Pattern: _Z (optional prefix digits) (length) (identifier)
    m = re.match(r"_Z\d*?(\d+)([A-Za-z_]\w*)", mangled)
    if m:
        length = int(m.group(1))
        name   = m.group(2)
        return name[:length]  # trim to declared length
    return mangled


def _kernel_identifier(name: str) -> str:
    """Return the callable identifier from an NCU or ptxas kernel name."""
    return re.sub(r"\(.*$", "", name.strip())
