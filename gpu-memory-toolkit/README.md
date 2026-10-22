# GPU Memory Toolkit — Week 2

This Week 2 deliverable automates the profiling path:

```text
CUDA source → nvcc → executable → Nsight Compute → normalized JSON
```

It preserves the compiler log, raw Nsight Compute CSV/report, metadata, and
`profile_result.json` in one result directory. The JSON is deliberately a
normalized input for the Week 3 detector work; it does not diagnose kernels.

## Requirements

- NVIDIA driver and CUDA-capable GPU
- `nvcc`, `ncu`, and `nvidia-smi` available on `PATH`
- Python 3.10+; no third-party Python package is required for Week 2

The profiler detects the GPU architecture automatically. Pass `--arch` only to
override it; for example, use `sm_86` for an Ampere RTX 30-series GPU.

## Run

From this directory:

```bash
python3 profiler/profile.py benchmarks/global_access/bad_global.cu --arch sm_86
```

Useful options:

```bash
# Store artifacts somewhere explicit
python3 profiler/profile.py benchmarks/bank_conflict/bad_bank.cu --arch sm_86 --output results/bad_bank

# Compile and emit ptxas-derived JSON without executing Nsight Compute
python3 profiler/profile.py benchmarks/register_spill/bad_spill.cu --arch sm_86 --no-ncu

# Limit Nsight Compute to a kernel name substring
python3 profiler/profile.py benchmarks/global_access/bad_global.cu --arch sm_86 --kernel bad_stride_write
```

## Output

`profile_result.json` contains system details, compile/run status, resource
usage from ptxas, normalized global/shared/local memory metrics, missing metric
names, and paths to raw artifacts. A missing metric remains explicitly missing;
the toolkit never substitutes a made-up value.

Run the parser-only verification without a GPU:

```bash
python3 -m unittest discover -s tests -v
```
