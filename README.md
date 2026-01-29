# Sovereign PIO

**Personal Intelligent Operator -- Autonomous Agent Operating System**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Cloudhabil/spio/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version 1.618.0](https://img.shields.io/badge/version-1.618.0-gold.svg)](https://github.com/Cloudhabil/spio)
[![Ruff](https://img.shields.io/badge/linter-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy-lang.org/)

Sovereign PIO routes AI workloads to NPU, CPU, and GPU silicon using deterministic golden-ratio mathematics. All routing decisions are O(1), all energy conserved at 2PI, all capacity governed by 840 discrete Lucas states across 12 cognitive dimensions.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [The 12 Dimensions](#the-12-dimensions)
- [Extensions](#extensions)
- [Mathematical Foundation](#mathematical-foundation)
- [Goldbach Extensions](#goldbach-extensions)
- [IIAS Framework](#iias-framework)
- [Configuration](#configuration)
- [Project Statistics](#project-statistics)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Installation

### Requirements

- Python >= 3.10
- pip >= 21.0

### From Source

```bash
git clone https://github.com/Cloudhabil/spio.git
cd spio
pip install -e .
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

### Development Install

```bash
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "from sovereign_pio.constants import PHI; print(f'PHI = {PHI}')"
# PHI = 1.6180339887498949
```

---

## Quick Start

### Initialize the System

```python
from extensions.iias import create_iias

iias = create_iias()
result = iias.initialize()
print(result["status"])       # "initialized"
print(result["dimensions"])   # 12
print(result["total_states"]) # 840
```

### Route a Workload

```python
from extensions.iias import DimensionRouter

router = DimensionRouter()
result = router.route(100.0)  # 100 MB request

# Workload is decomposed across 12 dimensions
# and routed to NPU (D1-D4), CPU (D5-D8), GPU (D9-D12)
print(result["total_time_ms"])
print(result["routing"])
```

### INDS Routing (O(1) Silicon Classification)

```python
from extensions.iias import DimensionRouter

router = DimensionRouter()

# Digital-root based deterministic routing
router.route_by_inds(42)   # -> "NPU" (dr=6 maps to GPU... actually dr=6 -> GPU)
router.route_by_inds(100)  # -> "NPU" (dr=1)
router.route_by_inds(77)   # -> "CPU" (dr=5)
```

### Evidence Database

```python
from src.core.evidence_db import EvidenceDB

with EvidenceDB("cycle_001.db") as db:
    db.record_constants()
    axioms = db.verify_axioms()  # 6 axioms verified
    summary = db.finalize()
    print(f"Size: {db.size_kb():.1f} KB")
```

---

## Architecture

```
+---------------------------------------------------------------------+
|                        APPLICATION LAYER                             |
|  Skills (50+)  |  Agents  |  IIAS (125 apps)  |  Research           |
+---------------------------------------------------------------------+
|                         OS LAYER                                     |
|  Kernel  |  Memory  |  Filesystem  |  Shell  |  Users  |  Safety    |
+---------------------------------------------------------------------+
|                      MATH BRIDGE                                     |
|  Wormhole Engine  |  PHI Transform  |  O(1) Routing  |  Evidence DB |
+---------------------------------------------------------------------+
|                      SILICON LAYER                                   |
|  D1-D4 : NPU (7.35 GB/s)  |  D5-D8 : CPU (26.0 GB/s)              |
|  D9-D12: GPU (12.0 GB/s)  |  SSD: 2.8 GB/s                        |
+---------------------------------------------------------------------+
```

All bandwidth values are hardware-measured (RTX 4070 SUPER, Intel AI Boost, DDR5, NVMe).

---

## The 12 Dimensions

Each dimension has a Lucas-number capacity and is mapped to a silicon layer.

| Dimension | Name          | Lucas States | Silicon | Brahim Weight |
|-----------|---------------|:------------:|---------|---------------|
| D1        | Perception    | 1            | NPU     | L(1) x B(1)  |
| D2        | Attention     | 3            | NPU     | L(2) x B(2)  |
| D3        | Security      | 4            | NPU     | L(3) x B(3)  |
| D4        | Stability     | 7            | NPU     | L(4) x B(4)  |
| D5        | Compression   | 11           | CPU     | L(5) x B(5)  |
| D6        | Harmony       | 18           | CPU     | L(6) x B(6)  |
| D7        | Reasoning     | 29           | CPU     | L(7) x B(7)  |
| D8        | Prediction    | 47           | CPU     | L(8) x B(8)  |
| D9        | Creativity    | 76           | GPU     | L(9) x B(9)  |
| D10       | Wisdom        | 123          | GPU     | L(10) x B(10)|
| D11       | Integration   | 199          | GPU     | L(11) x B(10)|
| D12       | Unification   | 322          | GPU     | L(12) x B(10)|

**Total capacity: 840 states** (sum of Lucas sequence L(1)..L(12))

**Brahim Numbers:** (27, 42, 60, 75, 97, 117, 139, 154, 172, 187) -- mirror pairs sum to 214, center 107.

---

## Extensions

Sovereign PIO ships with 28 extensions organized in three layers.

### OS Layer (10 extensions)

| Extension      | Module                   | Purpose                              |
|----------------|--------------------------|--------------------------------------|
| `kernel`       | `extensions.kernel`      | Process scheduling, worker pools     |
| `memory`       | `extensions.memory`      | Allocation tracking, budget control  |
| `filesystem`   | `extensions.filesystem`  | Virtual filesystem, artifact storage |
| `boot`         | `extensions.boot`        | Genesis initialization sequence      |
| `drivers`      | `extensions.drivers`     | 12-dimension to silicon routing      |
| `shell`        | `extensions.shell`       | Terminal interface, REPL             |
| `users`        | `extensions.users`       | Permissions, sandboxing              |
| `safety`       | `extensions.safety`      | Circuit breakers, failsafe triggers  |
| `networking`   | `extensions.networking`  | Protocol handlers                    |
| `gateway`      | `extensions.gateway`     | API and substrate access             |

### Intelligence Layer (10 extensions)

| Extension      | Module                     | Purpose                            |
|----------------|----------------------------|------------------------------------|
| `brahims_laws` | `extensions.brahims_laws`  | 6 Laws engine, elliptic curve analysis |
| `skills`       | `extensions.skills`        | 50+ autonomous skill definitions   |
| `agents`       | `extensions.agents`        | Multi-agent orchestration          |
| `iias`         | `extensions.iias`          | 125 infrastructure applications    |
| `cognitive`    | `extensions.cognitive`     | MetaCortex, alignment primitives   |
| `reflexes`     | `extensions.reflexes`      | System 1 reflexive triggers        |
| `research`     | `extensions.research`      | BSD conjecture, Hodge automation   |
| `wormhole`     | `extensions.wormhole`      | PHI transform engine               |
| `physics`      | `extensions.physics`       | Cosmology, Yang-Mills models       |
| `gpia_wormhole`| `extensions.gpia_wormhole` | Instant knowledge retrieval        |

### Infrastructure Layer (8 extensions)

| Extension   | Module                  | Purpose                         |
|-------------|-------------------------|---------------------------------|
| `bridges`   | `extensions.bridges`    | Cross-extension communication   |
| `boa_sdks`  | `extensions.boa_sdks`   | External SDK integrations       |
| `dashboard` | `extensions.dashboard`  | Monitoring and observability UI |
| `psi`       | `extensions.psi`        | Browser and APK interfaces      |
| `scripts`   | `extensions.scripts`    | Automation scripts              |
| `modes`     | `extensions.modes`      | Operational mode switching      |
| `budget`    | `extensions.budget`     | Resource budgeting              |
| `core`      | `extensions.core`       | Shared core utilities           |

---

## Mathematical Foundation

All computations are deterministic via Brahim's Calculator.

### Constants

```python
PHI   = (1 + sqrt(5)) / 2  # 1.6180339887498949  Golden ratio
ALPHA = PHI                 # 1.6180339887498949  Creation
OMEGA = 1 / PHI            # 0.6180339887498949  Return
BETA  = 1 / PHI**3         # 0.2360679774997897  Security
GAMMA = 1 / PHI**4         # 0.1458980337503155  Damping
```

### Core Functions

```python
def D(x):       return -log(x) / log(PHI)       # Dimension from value
def Theta(x):   return 2 * PI * x                # Phase from value
def Energy(x):  return PHI**D(x) * Theta(x)      # Always = 2*PI
def x_from_D(d): return 1 / PHI**d               # Inverse mapping
```

### Bandwidth Saturation (Measured)

```python
BW(N) = BW_max * (1 - exp(-N / k))

# NPU: BW_max = 7.35 GB/s, k = 1.64 (k ~ PHI)
# GPU: BW_max = 12.0 GB/s, k = 0.36
# CPU: BW_max = 26.0 GB/s, k = 0.90
```

### Brahim's 6 Laws

| # | Law                  | Formula                          |
|---|----------------------|----------------------------------|
| 1 | Brahim Conjecture    | Sha ~ Im(tau)^(2/3)             |
| 2 | Arithmetic Reynolds  | Rey = N / (Tam * Omega)          |
| 3 | Phase Transition     | Rey_c in [10, 30]                |
| 4 | Dynamic Scaling      | Sha_max ~ Rey^(5/12)            |
| 5 | Cascade Law          | Var(log Sha) ~ p^(-1/4)         |
| 6 | Consistency          | 2/3 = 5/12 + 1/4                |

---

## Goldbach Extensions

Six validated extensions derived from universal structure research (495,001 even integers verified).

| Extension                 | Description                                         |
|---------------------------|-----------------------------------------------------|
| **INDS Routing**          | O(1) digital-root silicon classifier, 42 pair types |
| **Lucas Partition**       | Multi-dimension occupancy guarantee (>= 4 dims)     |
| **Evidence DB**           | SQLite audit trail per SPIO cycle, 6 axiom checks   |
| **PHI-PI Gap Tolerance**  | (322*PI - 1000)/1000 = 1.159% explicit error margin |
| **2/3 Closure Budget**    | Productive (2/3) vs structural (1/3) resource split  |
| **Bandwidth Phi-Density** | phi^(-D_eff(n)) = 1/ln(n) identity                  |

### Usage

```python
from extensions.iias import DimensionRouter

router = DimensionRouter()

# INDS routing: digital root -> silicon
silicon = router.route_by_inds(42)  # O(1)

# Lucas partition: occupancy analysis
occ = router.lucas_partition_occupancy(100.0)
print(occ["occupied_dimensions"])  # >= 4 guaranteed

# 2/3 closure: productive budget
budget = router.productive_budget(1000.0)
print(budget["productive"])   # 666.67
print(budget["structural"])   # 333.33

# Real-time validation with phi-pi tolerance
rt = router.can_achieve_realtime_validated(10.0)
print(rt["feasible"])         # True/False
print(rt["tolerance_used"])   # 0.01159...
```

---

## IIAS Framework

Intelligent Infrastructure as a Service -- 125 deterministic applications across 13 categories.

| Category         | Apps | Examples                                    |
|------------------|:----:|---------------------------------------------|
| Foundation       | 5    | dimension_router, genesis_controller        |
| Infrastructure   | 10   | auto_scaler, load_balancer, cost_optimizer  |
| Edge             | 10   | edge_ai_router, battery_manager            |
| AI/ML            | 10   | inference_router, attention_allocator       |
| Security         | 10   | threat_classifier, anomaly_detector         |
| Business         | 10   | resource_allocator, task_scheduler          |
| Data             | 10   | data_tiering, cache_invalidator             |
| IoT              | 10   | device_router, telemetry_collector          |
| Communication    | 10   | message_router, protocol_selector           |
| Developer        | 10   | build_optimizer, feature_flagger            |
| Scientific       | 10   | simulation_router, hypothesis_ranker        |
| Personal         | 10   | focus_manager, habit_tracker                |
| Finance          | 10   | portfolio_balancer, risk_calculator         |

```python
from extensions.iias import AppRegistry

registry = AppRegistry()
stats = registry.stats()
print(f"Total: {stats['total']}")
print(f"Done: {stats['done']}")
print(f"Categories: {len(stats['categories'])}")
```

---

## Configuration

### pyproject.toml

```toml
[project]
name = "sovereign-pio"
version = "1.618.0"
requires-python = ">=3.10"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
```

### Environment Variables

| Variable         | Default | Description                   |
|------------------|---------|-------------------------------|
| `SPIO_LOG_LEVEL` | `INFO`  | Logging verbosity             |
| `SPIO_DATA_DIR`  | `data/` | Data and cache directory      |
| `SPIO_GPU`       | `auto`  | GPU device selection          |

### Optional Dependencies

| Group  | Command                       | Includes                    |
|--------|-------------------------------|-----------------------------|
| `dev`  | `pip install -e ".[dev]"`     | pytest, ruff, mypy          |
| `gpu`  | `pip install -e ".[gpu]"`     | torch, transformers         |
| `full` | `pip install -e ".[full]"`    | All optional dependencies   |

---

## Project Statistics

| Metric             | Value     |
|--------------------|-----------|
| Python files       | 149       |
| Lines of code      | 54,300    |
| Extensions         | 28        |
| IIAS applications  | 125       |
| Cognitive dims     | 12        |
| Total Lucas states | 840       |
| Goldbach extensions| 6         |
| Lint (ruff)        | Passing   |
| Type check (mypy)  | Passing   |

---

## Documentation

| Document                                  | Description                            |
|-------------------------------------------|----------------------------------------|
| [USAGE.md](USAGE.md)                      | Comprehensive usage guide              |
| [CLAUDE.md](CLAUDE.md)                    | AI assistant integration instructions  |
| [examples/](examples/)                    | Code examples and notebooks            |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/description`
3. Run lint and type checks:
   ```bash
   ruff check .
   mypy src/ extensions/ --ignore-missing-imports
   ```
4. Commit with a conventional message: `feat(scope): description`
5. Push and open a Pull Request

All contributions must pass `ruff` and `mypy` checks before merge.

---

## License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Sovereign PIO Team.

---

## Citation

```bibtex
@software{sovereign_pio_2026,
  title   = {Sovereign PIO: Autonomous Agent Operating System},
  author  = {Oulad Brahim, Elias},
  year    = {2026},
  url     = {https://github.com/Cloudhabil/spio},
  version = {1.618.0},
  license = {MIT}
}
```
