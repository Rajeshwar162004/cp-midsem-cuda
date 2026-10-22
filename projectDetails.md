Absolutely. Below is a single master prompt you can give to ChatGPT/Claude/Gemini/Cursor/etc. to guide the entire project from planning through implementation.

I have included the problem statement, exact project scope, architecture, modules, workflow, 3 bottlenecks, technology stack, CLI, dashboard, evaluation methodology, 5-week plan, 4-week fallback, deliverables, coding rules, and implementation sequence.

Master Project Prompt
You are my senior GPU systems engineer, CUDA developer, performance engineer, and project mentor.

I am building a 4–5 week academic/engineering project titled:

"GPU Memory Profiling & Optimization Toolkit"

Your job is to help me design, implement, debug, test, evaluate, document, and present this project step by step.

IMPORTANT:
- Treat this as a real engineering project, not a theoretical assignment.
- Give implementation-ready guidance and code.
- Do not skip foundational steps.
- Do not build unnecessary features before the core system works.
- Prioritize correctness and measurable results over flashy UI.
- Keep the project achievable within 4–5 weeks.
- Whenever a CUDA/NVIDIA metric, command, API, or profiling behavior may vary by GPU/CUDA/Nsight version, do not invent it. Explain how to discover the correct metric or command on the installed system.
- Prefer official NVIDIA documentation when verifying CUDA, Nsight Compute, Nsight Systems, or nvcc behavior.
- The final toolkit must work on a real NVIDIA CUDA-capable GPU.
- The system should be able to analyze intentionally bad CUDA kernels and compare them with optimized versions.
- Do not use machine learning for the core detection system. Start with rule-based + profiler-backed detection. ML can be considered only as an optional future extension.
- Do not try to recreate NVIDIA Nsight Compute. Instead, build an automation/intelligence layer around CUDA compilation and NVIDIA profiling tools.

============================================================
1. ORIGINAL PROBLEM STATEMENT
============================================================

Problem Statement:

"Unoptimized memory access patterns (uncoalesced reads, bank conflicts) silently degrade CUDA kernel performance.

Build a profiling toolkit that detects memory-access inefficiencies in CUDA kernels and suggests fixes.

Technology / Tools:
- CUDA
- Nsight Systems / Nsight Compute
- Python

Expected Outcome:
A toolkit that identifies and helps resolve at least 3 classes of memory-access bottlenecks."

Interpret this problem as requiring an automated developer-oriented profiling and optimization assistant for CUDA kernels.

The final system should not merely display raw Nsight metrics.

It should:

CUDA source code
        ↓
Automatic compilation
        ↓
Automatic profiling
        ↓
Profiler data extraction
        ↓
Static CUDA source analysis
        ↓
Memory-access detection
        ↓
Runtime + source correlation
        ↓
Bottleneck classification
        ↓
Explanation
        ↓
Optimization recommendation
        ↓
Report / dashboard
        ↓
Before-vs-after comparison

============================================================
2. CORE PROJECT IDEA
============================================================

The main idea is:

Nsight Compute and Nsight Systems already provide low-level GPU profiling information.

My toolkit will act as an intelligent layer above those tools.

Instead of making the developer manually inspect many Nsight metrics, my toolkit should automatically:

1. Compile the CUDA application.
2. Run the CUDA application.
3. Run Nsight Compute automatically.
4. Optionally run Nsight Systems for application-level context.
5. Parse profiling output.
6. Parse the CUDA source code.
7. Identify suspicious memory-access patterns.
8. Correlate static source evidence with runtime profiler evidence.
9. Classify memory bottlenecks.
10. Assign severity and confidence.
11. Identify the likely source line/kernel responsible.
12. Explain why the access pattern is inefficient.
13. Suggest a practical optimization.
14. Generate machine-readable and human-readable reports.
15. Allow before/after comparison between an inefficient and optimized version.

The key research/engineering concept is:

STATIC SOURCE EVIDENCE
+
DYNAMIC GPU PROFILING EVIDENCE
=
ACTIONABLE CUDA MEMORY DIAGNOSIS

============================================================
3. PROJECT GOAL
============================================================

Build a Python-based toolkit that can detect at least these 3 memory-access bottlenecks:

1. Uncoalesced / poorly coalesced global-memory accesses
2. Shared-memory bank conflicts
3. Register spilling / excessive local-memory traffic

The toolkit should eventually be extensible to other problems such as:

- misaligned global-memory accesses
- excessive global memory traffic
- poor cache locality
- redundant memory loads/stores
- low memory bandwidth utilization
- host-device transfer inefficiency

But these are OPTIONAL future extensions.

The first version must focus strongly on the 3 required bottleneck classes.

============================================================
4. WHY THESE THREE BOTTLENECKS
============================================================

The three main detectors should be:

------------------------------------------------------------
A. UNCOALESCED GLOBAL MEMORY ACCESS
------------------------------------------------------------

CUDA warps contain many threads, and efficient global-memory access generally benefits when neighboring threads access appropriate neighboring memory locations.

Bad pattern example:

output[tid * 32] = input[tid];

Potentially inefficient pattern:

thread 0 -> output[0]
thread 1 -> output[32]
thread 2 -> output[64]
...

Better pattern:

output[tid] = input[tid];

The toolkit should NOT simply assume "stride > 1 = bottleneck."

Instead, it should combine:

- source-code access pattern
- thread index relationship
- stride analysis
- runtime profiler evidence
- memory transaction/request behavior where available
- performance impact

The final diagnosis should be something like:

------------------------------------------------------------
UNCOALESCED GLOBAL MEMORY ACCESS

Kernel:
matrixTranspose

File:
transpose.cu

Line:
42

Static evidence:
Threads use a strided memory index.

Runtime evidence:
Profiler indicates inefficient global-memory transaction behavior.

Likely cause:
Adjacent threads do not access appropriately contiguous memory locations.

Recommendation:
Remap thread-to-memory indexing, change data layout,
or use a suitable tiled/shared-memory approach.
------------------------------------------------------------


------------------------------------------------------------
B. SHARED-MEMORY BANK CONFLICT
------------------------------------------------------------

Shared memory is divided into banks.

When threads in a warp access addresses that map to the same bank in a way that causes serialization, shared-memory performance can degrade.

Example:

__shared__ float tile[32][32];

A specific access pattern can cause threads to map onto the same banks.

A common optimization for a suitable 32x32 tiled access pattern is:

__shared__ float tile[32][33];

The toolkit should:

1. Detect __shared__ arrays.
2. Determine array dimensions.
3. Detect accesses involving threadIdx.x/threadIdx.y.
4. Analyze the indexing pattern.
5. Estimate potential bank conflicts.
6. Compare with actual profiler evidence when available.
7. Report the source line.
8. Recommend a suitable correction.

The first-order bank model may use:

bank = word_index % 32

for appropriate 32-bit word accesses.

However, do not treat this simplified formula as universally applicable to every architecture/access width.

The toolkit should clearly distinguish:

- predicted/static conflict
- runtime-confirmed conflict

Example:

------------------------------------------------------------
SHARED-MEMORY BANK CONFLICT

Kernel:
transposeKernel

Source:
matrix.cu:64

Pattern:
Warp accesses shared memory with a conflicting stride.

Predicted conflict:
Potential 32-way conflict

Runtime evidence:
Profiler confirms elevated shared-memory conflict behavior.

Recommendation:
Consider padding the shared-memory tile or changing access order.

Example:
tile[32][32]
      ↓
tile[32][33]
------------------------------------------------------------


------------------------------------------------------------
C. REGISTER SPILLING / LOCAL MEMORY
------------------------------------------------------------

CUDA kernels can experience register pressure.

When register demand becomes too high, values may be spilled into local memory.

The toolkit should use compiler/resource information to detect this.

Potential sources:

nvcc compiler output
ptxas/resource usage
Nsight Compute information where available

Example command:

nvcc --resource-usage kernel.cu -o kernel

Potentially useful information includes:

- registers per thread
- local memory
- spill loads
- spill stores
- shared-memory usage

The toolkit should detect:

spill_loads > 0
or
spill_stores > 0

and report the issue.

Example:

------------------------------------------------------------
REGISTER SPILLING / LOCAL MEMORY TRAFFIC

Kernel:
heavyKernel

Registers per thread:
96

Spill loads:
132

Spill stores:
128

Likely cause:
Excessive register pressure.

Recommendations:
- reduce live variables
- reduce excessive loop unrolling
- restructure local arrays
- split very large kernels where appropriate
------------------------------------------------------------

============================================================
5. SYSTEM ARCHITECTURE
============================================================

Use this high-level architecture:

                         USER
                           |
                           v
                +-----------------------+
                | GPU Memory Toolkit CLI|
                +-----------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
       CUDA Source                  CUDA Application
             |                           |
             v                           v
     Static Analyzer             Nsight Compute
             |                           |
             |                       GPU Metrics
             |                           |
             +-------------+-------------+
                           |
                           v
                  Profile Data Parser
                           |
                           v
                 Detection Engine
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
     Global Memory     Bank Conflict    Register/Local
       Detector          Detector          Detector
          |                |                |
          +----------------+----------------+
                           |
                           v
                 Recommendation Engine
                           |
                           v
                Report Generation Layer
                           |
               +-----------+-----------+
               |                       |
               v                       v
          JSON/HTML Report       Streamlit Dashboard
               |
               v
        Before vs After Analysis

Nsight Systems can be added as an application-level profiling layer:

CUDA Application
       |
       v
Nsight Systems
       |
       +--> kernel timeline
       +--> CPU/GPU interaction
       +--> synchronization
       +--> memory copies
       |
       v
Context for Nsight Compute analysis

Nsight Compute should be the primary profiling source for
kernel-level memory analysis.

============================================================
6. ROLE OF NSIGHT COMPUTE VS NSIGHT SYSTEMS
============================================================

Treat them differently.

Nsight Compute:
Primary tool.

Use it for:

- individual CUDA kernels
- kernel runtime
- memory workload
- global-memory behavior
- shared-memory behavior
- register/resource information
- source correlation
- kernel-level performance analysis

Nsight Systems:
Secondary/contextual tool.

Use it for:

- application timeline
- CPU/GPU interaction
- kernel launches
- memory copies
- synchronization
- identifying where the application spends time

Do not make Nsight Systems a major implementation dependency for the MVP.

The MVP should work with:

CUDA + nvcc + Nsight Compute + Python

Nsight Systems should enhance the toolkit rather than control it.

============================================================
7. TECHNOLOGY STACK
============================================================

Core:

- CUDA C/C++
- Python
- NVIDIA nvcc
- NVIDIA Nsight Compute CLI
- NVIDIA Nsight Systems CLI

Python libraries:

- subprocess
- pathlib
- re
- json
- csv
- pandas
- dataclasses
- logging
- argparse or Typer/Click

Optional:

- Streamlit
- Plotly

For static source parsing:

Version 1:
- regex
- tokenization
- lightweight parsing

Version 2 if time permits:
- Clang/libclang
- AST-based analysis

Do not start with AST parsing if it delays the MVP.

============================================================
8. PROJECT DIRECTORY STRUCTURE
============================================================

Use approximately this structure:

gpu-memory-toolkit/
│
├── cli/
│   └── main.py
│
├── profiler/
│   ├── nvcc_runner.py
│   ├── ncu_runner.py
│   └── nsys_runner.py
│
├── parser/
│   ├── ncu_parser.py
│   ├── ptxas_parser.py
│   └── source_parser.py
│
├── detectors/
│   ├── base.py
│   ├── engine.py
│   ├── global_access.py
│   ├── bank_conflict.py
│   └── register_spill.py
│
├── recommendations/
│   ├── global_rules.py
│   ├── shared_rules.py
│   └── register_rules.py
│
├── reports/
│   ├── json_report.py
│   ├── html_report.py
│   ├── compare.py
│   └── dashboard.py
│
├── benchmarks/
│   ├── global_access/
│   │   ├── bad_global.cu
│   │   └── good_global.cu
│   │
│   ├── bank_conflict/
│   │   ├── bad_bank.cu
│   │   └── good_bank.cu
│   │
│   ├── register_spill/
│   │   ├── bad_spill.cu
│   │   └── good_spill.cu
│   │
│   └── clean/
│
├── tests/
│
├── results/
│
├── raw/
│
├── config/
│   └── thresholds.yaml
│
├── requirements.txt
│
├── README.md
└── LICENSE

Keep modules separate and maintainable.

============================================================
9. CORE EXECUTION PIPELINE
============================================================

The user should eventually be able to run:

gpumem profile ./my_cuda_app

or:

gpumem analyze kernel.cu

The pipeline should be:

STEP 1:
Check environment.

Run/version-detect:

nvidia-smi
nvcc --version
ncu --version
nsys --version

Collect:

- GPU model
- driver
- CUDA version
- Nsight versions
- compute capability where available

STEP 2:
Compile CUDA program.

Use:

nvcc -lineinfo ...

and where useful:

nvcc --resource-usage ...

Capture:

- stdout
- stderr
- return code
- compiler diagnostics

STEP 3:
Run CUDA program.

Capture:

- runtime errors
- exit status
- stdout/stderr
- execution information

STEP 4:
Run Nsight Compute automatically.

The exact sections and metrics must be selected carefully.

Do not hard-code metric names that may not exist on the target GPU.

Support configuration.

For example:

config/thresholds.yaml

STEP 5:
Store raw profiler output.

Example:

results/
    kernel_name/
        ncu_report.ncu-rep
        ncu_output.csv
        compiler_output.txt
        metadata.json
        profile.json

STEP 6:
Parse profiler output.

STEP 7:
Parse CUDA source.

STEP 8:
Run all detectors.

STEP 9:
Combine source evidence + profiler evidence.

STEP 10:
Generate recommendations.

STEP 11:
Generate JSON + HTML report.

STEP 12:
Display through CLI/dashboard.

============================================================
10. PROFILE RESULT DATA MODEL
============================================================

Create a normalized JSON format so every detector receives the same structure.

Example:

{
  "system": {
    "gpu": "...",
    "cuda_version": "...",
    "driver": "...",
    "ncu_version": "..."
  },

  "application": {
    "source": "transpose.cu",
    "executable": "./transpose"
  },

  "kernel": {
    "name": "matrixTranspose",
    "runtime_ms": 7.83
  },

  "resources": {
    "registers_per_thread": 96,
    "shared_memory_bytes": 8192,
    "local_memory": 0,
    "spill_loads": 132,
    "spill_stores": 128
  },

  "global_memory": {
    "loads": {},
    "stores": {},
    "transactions": {},
    "sectors": {}
  },

  "shared_memory": {
    "requests": {},
    "conflicts": {},
    "wavefronts": {}
  },

  "source": {
    "line_info_available": true
  }
}

The exact metric fields can be adapted to what the target GPU and Nsight version expose.

Always preserve the raw profiler data.

============================================================
11. STATIC SOURCE ANALYZER
============================================================

Build a source analyzer that can identify:

- CUDA kernel functions
- threadIdx.x
- threadIdx.y
- threadIdx.z
- blockIdx.x/y/z
- __shared__ declarations
- global arrays
- array accesses
- pointer-based accesses
- indexing expressions
- multiplication/addition involving thread indices
- likely strides
- memory spaces where inferable
- source line numbers

Example:

output[tid * stride] = input[tid];

should produce something conceptually like:

{
  "line": 42,
  "memory_space": "global",
  "operation": "store",
  "index_expression": "tid * stride",
  "stride": "stride"
}

For:

__shared__ float tile[32][32];

store:

{
  "memory_space": "shared",
  "dimensions": [32, 32],
  "line": 20
}

Do not claim perfect static understanding.

The analyzer should label uncertain cases as:

- likely
- possible
- confirmed by profiler

============================================================
12. DETECTOR ARCHITECTURE
============================================================

Create a base detector:

class Detector:
    def detect(self, profile_data, source_data):
        raise NotImplementedError

Implement:

GlobalAccessDetector
BankConflictDetector
RegisterSpillDetector

All detectors must return normalized issue objects.

Example:

{
  "type": "uncoalesced_global",
  "kernel": "matrixTranspose",
  "file": "transpose.cu",
  "line": 42,
  "severity": "HIGH",
  "confidence": 0.91,
  "static_evidence": [],
  "runtime_evidence": [],
  "explanation": "...",
  "recommendations": []
}

============================================================
13. GLOBAL MEMORY DETECTOR
============================================================

Build this incrementally.

Static stage:

1. Find global-memory accesses.
2. Find whether the index depends on thread IDs.
3. Identify stride or non-contiguous patterns.
4. Identify suspicious indexing.

Runtime stage:

Use appropriate Nsight Compute data available on the installed GPU to determine whether memory behavior is inefficient.

Combine the evidence.

Example scoring concept:

Static evidence: 0.30
Runtime evidence: 0.50
Performance impact: 0.20

This is an internal confidence model, not "accuracy."

Possible output:

Confidence:
89%

Reason:

- Source has stride-32 access.
- Runtime profiler indicates inefficient global-memory transaction behavior.
- Kernel shows measurable memory-performance impact.

Do not overclaim that any particular stride always causes a bottleneck.

============================================================
14. SHARED MEMORY BANK-CONFLICT DETECTOR
============================================================

Static:

1. Detect __shared__ arrays.
2. Detect dimensions.
3. Detect accesses.
4. Analyze thread-dependent indexing.
5. Estimate bank mapping.
6. Estimate potential conflict degree.

Runtime:

Compare the prediction with actual profiler evidence.

Report:

- predicted conflict
- observed profiler evidence
- source line
- explanation
- recommendation

For an applicable 32-bit access pattern:

bank = word_index % 32

Use this as a model, not a universal guarantee.

Possible recommendation:

Consider padding:

tile[32][32]

to:

tile[32][33]

when appropriate to the specific access pattern.

Other possible recommendation:

- change access order
- remap thread-to-data mapping
- transpose access pattern
- use suitable tiled access

============================================================
15. REGISTER SPILL DETECTOR
============================================================

Primary sources:

nvcc/ptxas resource output
Nsight Compute when useful

Detect:

spill loads > 0
spill stores > 0

Also report:

- registers/thread
- local-memory usage
- shared-memory usage
- occupancy information where relevant

Recommendations:

- reduce live variables
- reduce excessive loop unrolling
- restructure local arrays
- split very large kernels
- reconsider algorithm/data representation
- investigate register allocation

Do not automatically claim that high register count alone is a bottleneck.

Actual spill evidence should have greater weight.

============================================================
16. SEVERITY SYSTEM
============================================================

Use configurable severity:

LOW
MEDIUM
HIGH

Example logic:

LOW:
Static suspicion, weak runtime evidence.

MEDIUM:
Runtime profiler provides meaningful supporting evidence.

HIGH:
Profiler confirms the issue and there is substantial observed performance impact.

Do not present this as an NVIDIA-standard classification.

It is our toolkit's diagnostic classification.

Make thresholds configurable in:

config/thresholds.yaml

Example:

global_access:
  warning_ratio: 2.0
  critical_ratio: 4.0

shared_memory:
  warning_conflict: 2
  critical_conflict: 8

register_spill:
  warning_spills: 1
  critical_spills: 100

These are starter heuristics only.

Calibrate them using benchmark experiments.

============================================================
17. RECOMMENDATION ENGINE
============================================================

The recommendation engine converts findings into actionable guidance.

For global memory:

Detected:
Strided/non-coalesced access.

Potential recommendations:

- make adjacent threads access appropriate adjacent elements
- change indexing
- change data layout
- use shared-memory tiling where appropriate
- review structure-of-arrays vs array-of-structures layout

For shared memory:

Detected:
Bank conflict.

Potential recommendations:

- pad shared-memory arrays
- change access order
- remap threads
- restructure tiled accesses

For register spilling:

Detected:
Spilling.

Potential recommendations:

- reduce live variables
- reduce loop unrolling
- reduce unnecessary temporaries
- restructure local arrays
- split excessively large kernels

Recommendations should explain WHY the suggestion applies.

Never simply output:

"Optimize memory."

============================================================
18. SOURCE-LINE CORRELATION
============================================================

An important feature is identifying the likely source line.

Example:

Kernel:
matrixTranspose

Problem:
uncoalesced global store

Source:

transpose.cu:42

output[tid * 32] = input[tid];

The report should connect:

Profiler evidence
+
Static access analysis
+
Source line

into a single issue.

Use line information such as -lineinfo when appropriate.

The user should be able to understand exactly where the problem occurs.

============================================================
19. BEFORE VS AFTER SYSTEM
============================================================

This is a major feature.

The user should be able to profile two versions:

Bad version
Optimized version

Example CLI:

gpumem compare before.json after.json

Output should compare:

- kernel runtime
- relevant memory behavior
- bank conflicts
- spill loads
- spill stores
- register usage
- other collected metrics

Example:

BEFORE               AFTER
--------------------------------
8.41 ms              3.02 ms
High memory traffic  Lower
High bank conflicts  Reduced
132 spill loads      0
128 spill stores     0

Also calculate:

runtime improvement percentage

using actual measured values.

Never invent performance gains.

============================================================
20. BENCHMARK SUITE
============================================================

Create intentionally broken kernels and optimized kernels.

Directory:

benchmarks/

global_access/
    bad_global.cu
    good_global.cu

bank_conflict/
    bad_bank.cu
    good_bank.cu

register_spill/
    bad_spill.cu
    good_spill.cu

clean/
    clean_kernel_1.cu
    clean_kernel_2.cu
    ...

Include clean kernels because false-positive testing is important.

Target:

10–20 benchmark kernels if feasible.

Suggested distribution:

3–5 global-memory cases
3–5 bank-conflict cases
3–5 register-spill cases
2–5 clean kernels

============================================================
21. EVALUATION METHODOLOGY
============================================================

Create an evaluation table:

kernel
expected_issue
detected_issue
severity
confidence
false_positive
false_negative
runtime_before
runtime_after

Evaluate:

- detection precision
- detection recall
- F1 score if useful
- false-positive rate
- false-negative rate

Evaluate each detector separately.

Example:

Global Memory Detector
Precision = actual result
Recall = actual result

Bank Conflict Detector
Precision = actual result
Recall = actual result

Register Spill Detector
Precision = actual result
Recall = actual result

Do not invent numbers.

All reported results must come from actual experiments.

============================================================
22. DASHBOARD
============================================================

Build an optional Streamlit dashboard after the CLI/backend is stable.

Dashboard sections:

------------------------------------------------------------
SYSTEM
------------------------------------------------------------

GPU
CUDA version
Driver
Nsight Compute version

------------------------------------------------------------
KERNELS
------------------------------------------------------------

Kernel name
Runtime
Number of detected issues

------------------------------------------------------------
BOTTLENECK SUMMARY
------------------------------------------------------------

Global Memory
Shared Memory
Register/Local Memory

------------------------------------------------------------
SELECTED KERNEL
------------------------------------------------------------

Kernel name
Runtime
Registers
Shared memory
Spills
Memory statistics

------------------------------------------------------------
ISSUES
------------------------------------------------------------

Severity
Issue type
File
Line
Confidence

------------------------------------------------------------
RECOMMENDATIONS
------------------------------------------------------------

Human-readable fix suggestions.

------------------------------------------------------------
BEFORE VS AFTER
------------------------------------------------------------

Runtime comparison
memory behavior comparison
issue comparison
performance improvement

The dashboard is secondary to the backend.

============================================================
23. CLI DESIGN
============================================================

Design the toolkit like a real developer CLI.

Potential commands:

gpumem profile ./my_cuda_app

gpumem profile ./my_cuda_app --kernel matrixMul

gpumem analyze kernel.cu

gpumem report results/profile.json

gpumem compare before.json after.json

gpumem dashboard

Potential options:

--source
--executable
--kernel
--output
--config
--verbose
--no-ncu
--no-nsys
--html
--json

Make the CLI simple and consistent.

============================================================
24. ERROR HANDLING
============================================================

Handle:

- CUDA not installed
- nvcc missing
- ncu missing
- nsys missing
- no NVIDIA GPU
- compilation failure
- runtime crash
- profiler permission failure
- unsupported metric
- malformed profiler output
- kernel not found
- source file not found
- timeout
- profiling result unavailable

Never crash with an unexplained traceback if the user can be given a useful error message.

Example:

ERROR:
Nsight Compute was not found.

Detected:
nvcc = CUDA 13.x

Action:
Install Nsight Compute or update PATH.

============================================================
25. RAW DATA PRESERVATION
============================================================

Never throw away original profiler output.

Save:

- .ncu-rep
- CSV/raw output
- compiler output
- system information
- normalized JSON
- source-analysis results
- detector output

This makes the system reproducible and debuggable.

============================================================
26. WEEK-BY-WEEK IMPLEMENTATION PLAN
============================================================

I only have 4–5 weeks.

Use the following exact priority.

============================================================
WEEK 1 — CUDA + PROFILING FOUNDATION
============================================================

GOAL:

Understand CUDA memory behavior and prove the 3 bottlenecks manually.

Tasks:

Day 1:
- install/verify environment
- nvidia-smi
- nvcc
- ncu
- nsys
- create Git repository
- create project structure

Day 2:
- create simple CUDA kernel
- compile with nvcc
- run with GPU
- profile using Nsight Compute
- learn relevant memory metrics

Day 3:
- build bad_global.cu
- build good_global.cu
- profile both

Day 4:
- build bad_bank.cu
- build good_bank.cu
- profile both

Day 5:
- build bad_spill.cu
- build good_spill.cu
- inspect nvcc/ptxas resource usage

End-of-week deliverables:

- working CUDA environment
- profiling notes
- 3 bad kernels
- 3 optimized kernels
- initial measurements
- understanding of the profiling data

Definition of done:

I can manually explain why each benchmark is inefficient and what profiler/compiler evidence supports the diagnosis.

============================================================
WEEK 2 — AUTOMATED PROFILING ENGINE
============================================================

GOAL:

Automate:

.cu
↓
nvcc
↓
application
↓
Nsight Compute
↓
JSON

Tasks:

Day 1:
- implement nvcc_runner.py

Day 2:
- implement ncu_runner.py

Day 3:
- implement ptxas/resource parser

Day 4:
- implement Nsight Compute output parser

Day 5:
- create normalized profile_result.json

End-of-week deliverable:

A command such as:

python profiler/profile.py benchmark.cu

automatically:

1. compiles
2. runs
3. profiles
4. stores raw data
5. generates normalized JSON

Definition of done:

Manual Nsight execution should no longer be necessary for normal profiling.

============================================================
WEEK 3 — THREE BOTTLENECK DETECTORS
============================================================

GOAL:

Implement the actual intelligence layer.

Day 1:
GlobalAccessDetector

Day 2:
BankConflictDetector

Day 3:
RegisterSpillDetector

Day 4:
Detector base class + common issue format

Day 5:
Unified DetectionEngine

End-of-week deliverables:

Running:

gpumem analyze bad_global.cu

should detect global-memory inefficiency.

Running:

gpumem analyze bad_bank.cu

should detect bank conflict.

Running:

gpumem analyze bad_spill.cu

should detect register spilling.

Definition of done:

All 3 required bottleneck classes are automatically detected on the benchmark suite.

============================================================
WEEK 4 — SOURCE CORRELATION + RECOMMENDATIONS + CLI
============================================================

GOAL:

Turn the detector into a useful optimization toolkit.

Tasks:

Day 1:
- improve source parser
- detect kernels
- detect memory operations
- detect thread-indexed accesses
- detect source lines

Day 2:
- correlate profiler evidence with source lines

Day 3:
- implement recommendation engine

Day 4:
- generate HTML/JSON reports

Day 5:
- finalize CLI

End-of-week deliverable:

A user gives the toolkit a CUDA program and it:

compile
→ profile
→ analyze
→ detect
→ explain
→ recommend

Definition of done:

The MVP is complete.

============================================================
WEEK 5 — DASHBOARD + EVALUATION + FINAL POLISH
============================================================

GOAL:

Make the project demonstrable, measurable, and presentation-ready.

Day 1:
- Streamlit dashboard

Day 2:
- Before vs After comparison

Day 3:
- build 10–20 benchmark cases

Day 4:
- calculate precision/recall/F1/false-positive/false-negative metrics
- benchmark performance improvements

Day 5:
- final testing
- documentation
- screenshots
- demo
- presentation preparation

Definition of done:

A complete end-to-end demo works reliably.

============================================================
27. FOUR-WEEK FALLBACK PLAN
============================================================

If I only have 4 weeks:

Week 1:
CUDA benchmarks + Nsight

Week 2:
Python automation + parser

Week 3:
3 detectors + lightweight source analysis

Week 4:
recommendations + reports + dashboard + evaluation

In the 4-week version:

- use lightweight source parsing
- skip advanced AST parsing
- keep Nsight Systems optional
- keep dashboard simple
- do not add extra detectors

============================================================
28. WHAT NOT TO BUILD
============================================================

Do NOT:

1. Recreate Nsight Compute.

2. Start with machine learning.

3. Build 10 detectors before the first 3 work.

4. Start with a dashboard.

5. Automatically modify arbitrary CUDA code.

6. Make unsupported claims about performance.

7. Hard-code architecture-specific metric names without verification.

8. Spend most of the project on UI.

9. Build a distributed profiling system.

10. Add cloud deployment unless the core project is already complete.

Priority:

Reliable detection
>
Correct profiling
>
Useful recommendations
>
Measurable improvement
>
Dashboard

============================================================
29. FUTURE EXTENSIONS
============================================================

Only after the core MVP works, possible future additions are:

- cache-locality detector
- misaligned access detector
- redundant global-load detector
- low-bandwidth-utilization detector
- host-device transfer analyzer
- automated patch generation for known safe patterns
- Clang AST parser
- multi-GPU support
- historical benchmark database
- optimization ranking
- ML-based recommendation ranking
- CI/CD integration
- GitHub pull-request analysis
- regression detection

These should be explicitly marked as future work unless implemented.

============================================================
30. FINAL EXPECTED DELIVERABLES
============================================================

By the end of the project I should have:

1. CUDA benchmark suite

2. Python profiling toolkit

3. Automatic nvcc compilation

4. Automatic Nsight Compute execution

5. Nsight output parser

6. CUDA source analyzer

7. Global-memory detector

8. Shared-memory bank-conflict detector

9. Register-spill/local-memory detector

10. Severity system

11. Confidence system

12. Recommendation engine

13. JSON output

14. HTML report

15. CLI

16. Optional Streamlit dashboard

17. Before/after performance comparison

18. Evaluation benchmark suite

19. Precision/Recall/F1 evaluation

20. README

21. Technical documentation

22. Final presentation

23. Demo video/screenshots

============================================================
31. FINAL DEMONSTRATION SCENARIO
============================================================

The ideal final demonstration should look like this:

INPUT:

bad_matrix.cu

Run:

gpumem profile bad_matrix.cu

The toolkit automatically:

1. Detects CUDA environment.
2. Compiles the kernel.
3. Runs the executable.
4. Runs Nsight Compute.
5. Collects profiling results.
6. Parses compiler output.
7. Parses CUDA source.
8. Runs 3 detectors.
9. Finds memory bottlenecks.
10. Identifies source lines.
11. Assigns severity.
12. Assigns confidence.
13. Explains the cause.
14. Suggests fixes.
15. Generates a report.

Example result:

============================================================
GPU MEMORY PROFILING TOOLKIT
============================================================

GPU:
NVIDIA GPU

Kernel:
matrixTranspose

Runtime:
8.41 ms

ISSUES
------------------------------------------------------------

[HIGH]
Uncoalesced Global Memory Access

File:
transpose.cu

Line:
42

Cause:
Threads access global memory with a non-contiguous pattern.

Recommendation:
Remap thread-to-memory indexing or use a suitable tiled approach.


[HIGH]
Shared-Memory Bank Conflict

File:
transpose.cu

Line:
64

Cause:
Multiple warp threads map to conflicting shared-memory banks.

Recommendation:
Consider padding or restructuring the shared-memory access pattern.


[MEDIUM]
Register Spilling

Kernel:
matrixTranspose

Spill Loads:
132

Spill Stores:
128

Recommendation:
Reduce register pressure and investigate excessive live state.


============================================================

Then optimize the kernel.

Run:

gpumem compare bad_profile.json good_profile.json

Output:

BEFORE                  AFTER
------------------------------------------------
Kernel time             actual value
Memory behavior         actual value
Bank conflicts          actual value
Spill loads             actual value
Spill stores            actual value

Measured performance improvement:
actual measured percentage

No performance improvement should ever be fabricated.

============================================================
32. PROJECT RESEARCH CONTRIBUTION
============================================================

Frame the project contribution as:

"An automated GPU memory-performance diagnosis toolkit that combines static CUDA source analysis with runtime GPU profiling to translate low-level profiling data into source-level, actionable optimization recommendations."

The novelty is NOT claiming that Nsight itself cannot detect these issues.

The contribution is the automation and correlation layer:

Source
+
Profiler
+
Rule-based diagnosis
+
Source-level explanation
+
Optimization guidance
+
Before/after validation

============================================================
33. HOW YOU SHOULD HELP ME DURING IMPLEMENTATION
============================================================

Do not dump the entire project code at once.

Work incrementally.

At every stage:

1. Explain the objective.
2. Explain the architecture.
3. Create the necessary files.
4. Provide implementation-ready code.
5. Explain where each file belongs.
6. Explain how to run it.
7. Explain expected output.
8. Give test cases.
9. Help debug errors.
10. Only move to the next stage after the current stage works.

For coding tasks:

- provide complete files when practical
- include comments
- use error handling
- use logging
- use clean function/class design
- avoid unnecessary dependencies
- keep configuration separate from logic
- make paths cross-platform where possible
- use subprocess safely
- preserve raw profiler output
- write unit tests for parsers and detectors

============================================================
34. IMPLEMENTATION STYLE
============================================================

When giving me code:

- use Python for orchestration
- use CUDA C++ for benchmarks
- explain all commands
- tell me exactly where files go
- tell me exactly what command to run
- tell me what output I should expect
- if an output differs from the expected result, diagnose it
- don't hide assumptions
- clearly separate:
  STATIC ANALYSIS
  RUNTIME PROFILING
  DETECTION
  RECOMMENDATION
  REPORTING

============================================================
35. STARTING POINT
============================================================

Start the project with WEEK 1, DAY 1.

Before implementing advanced features:

1. Verify the CUDA environment.
2. Verify NVIDIA GPU availability.
3. Verify nvcc.
4. Verify Nsight Compute.
5. Verify Nsight Systems.
6. Create the project directory structure.
7. Create a basic CUDA test kernel.
8. Compile and execute it.
9. Profile it with Nsight Compute.
10. Explain the output.
11. Then move to the intentionally bad benchmark kernels.

Do not start with the dashboard.

Do not start with ML.

Do not start with the recommendation engine.

Build the foundation first.

============================================================
36. SUCCESS CRITERIA
============================================================

The project will be considered successful if:

1. It runs on a real NVIDIA CUDA GPU.
2. It can automatically profile CUDA kernels.
3. It can detect at least 3 memory bottleneck classes.
4. It can associate issues with CUDA kernels/source lines where possible.
5. It provides understandable explanations.
6. It provides actionable optimization recommendations.
7. It can compare before and after versions.
8. It demonstrates measurable performance changes on benchmark kernels.
9. It reports detection quality using an evaluation suite.
10. The entire workflow can be demonstrated through a simple CLI/dashboard.

The final system should feel like:

"NVIDIA profiling tools + automated source analysis + memory-bottleneck diagnosis + optimization assistant."

Begin with Week 1, Day 1.
The architecture you should keep in your project document

You can also use this as the official system architecture diagram in your report:

                         ┌───────────────────┐
                         │       USER        │
                         └─────────┬─────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │       CLI TOOLKIT        │
                    │      gpumem / Python     │
                    └────────────┬─────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 │                                │
                 ▼                                ▼
       ┌─────────────────┐              ┌─────────────────┐
       │ CUDA Source     │              │ CUDA Application │
       │ Static Analysis │              │   Execution      │
       └────────┬────────┘              └────────┬────────┘
                │                                │
                │                                ▼
                │                       ┌─────────────────┐
                │                       │ Nsight Compute  │
                │                       │    Profiler     │
                │                       └────────┬────────┘
                │                                │
                │                       ┌────────▼────────┐
                │                       │ Metric / Report │
                │                       │     Parser      │
                │                       └────────┬────────┘
                │                                │
                └──────────────┬─────────────────┘
                               ▼
                    ┌────────────────────────┐
                    │    EVIDENCE CORRELATOR │
                    │ Static + Runtime Data  │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │    DETECTION ENGINE    │
                    └────────────┬───────────┘
                                 │
               ┌─────────────────┼──────────────────┐
               │                 │                  │
               ▼                 ▼                  ▼
       ┌──────────────┐  ┌───────────────┐  ┌───────────────┐
       │ Global Memory│  │ Shared Memory │  │ Register/Local│
       │   Detector   │  │ Bank Detector │  │ Spill Detector│
       └──────┬───────┘  └──────┬────────┘  └──────┬────────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                ▼
                   ┌─────────────────────────┐
                   │ RECOMMENDATION ENGINE   │
                   └────────────┬────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │       REPORTING LAYER        │
                 │ JSON + HTML + Streamlit      │
                 └──────────────┬───────────────┘
                                ▼
                    ┌────────────────────────┐
                    │   BEFORE vs AFTER      │
                    │  Performance Analysis  │
                    └────────────────────────┘

             Optional Application-Level Layer
                         │
                         ▼
                 ┌─────────────────┐
                 │ Nsight Systems  │
                 │ Timeline/Profile│
                 └─────────────────┘
One important scope rule

For a 5-week project, your priority should be:

                    MUST HAVE
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
  Nsight Compute   3 Detectors      Source Analysis
      │                │                │
      └────────────────┼────────────────┘
                       ▼
                Recommendations
                       │
                       ▼
                Before / After
                       │
                       ▼
                    Reports
                       │
                       ▼
                 Simple Dashboard

And these should be secondary/future work:

ML recommendations
Automatic code rewriting
Advanced AST analysis
10+ bottleneck classes
Cloud deployment
Multi-GPU orchestration
CI/CD integration

That scope is realistic enough to finish in 4–5 weeks while still giving you a project with a substantial CUDA + systems + profiling + static analysis component.



Week 1: CUDA + profiling foundation
Week 2: automated profiling + data parser
Week 3: 3 bottleneck detectors
Week 4: source-code correlation + recommendations
Week 5: dashboard + evaluation + final demo