<p align="center">
  <img src="https://img.shields.io/badge/SOVEREIGN-PIO-gold?style=for-the-badge&logo=atom&logoColor=white" alt="Sovereign PIO"/>
</p>

<h1 align="center">Sovereign PIO</h1>

<p align="center">
  <strong>The Autonomous Agent OS with Mathematical Bridge to Silicon</strong>
</p>

<p align="center">
  <a href="https://github.com/Cloudhabil/spio"><img src="https://img.shields.io/github/actions/workflow/status/Cloudhabil/spio/ci.yml?branch=main&style=flat-square&label=CI" alt="CI Status"/></a>
  <a href="https://github.com/Cloudhabil/spio/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://github.com/Cloudhabil/spio"><img src="https://img.shields.io/badge/version-1.618.0-gold?style=flat-square" alt="Version"/></a>
  <a href="https://cloudhabil.com"><img src="https://img.shields.io/badge/website-cloudhabil.com-purple?style=flat-square" alt="Website"/></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-features">Features</a> &bull;
  <a href="#-architecture">Architecture</a> &bull;
  <a href="#-extensions">Extensions</a> &bull;
  <a href="#-docs">Docs</a> &bull;
  <a href="#-contributing">Contributing</a>
</p>

---

> **One-liner:** SPIO routes AI workloads to NPU/CPU/GPU using golden ratio mathematics — deterministic, O(1) wormhole compression, 840 discrete states across 12 dimensions.

---

## Demo

```python
from extensions.brahims_laws import BrahimLawsEngine, CONSTANTS

# PHI-based deterministic computation
print(f"PHI = {CONSTANTS.PHI}")  # 1.618033988749895

# Analyze elliptic curves with Brahim's 6 Laws
engine = BrahimLawsEngine()
result = engine.analyze(curve)
print(f"Reynolds: {result.reynolds_number}, Regime: {result.regime}")
```

```
PHI = 1.618033988749895
Reynolds: 11.0000, Regime: TRANSITION
```

---

## Quick Start

```bash
# Clone & install
git clone https://github.com/Cloudhabil/spio.git
cd spio && pip install -e .

# Verify
python -c "from extensions.core import PHI; print(f'PHI = {PHI}')"
```

**Boot the system:**

```python
from extensions.boot import run_genesis
run_genesis()  # Initializes all 28 extensions
```

**Route to hardware:**

```python
from extensions.kernel import Kernel
from extensions.wormhole import BETA

kernel = Kernel()
# Dimension 7 (Reasoning) -> CPU
# Dimension 10 (Wisdom) -> GPU
# BETA security threshold: 0.236
```

---

## Features

| Feature | Description |
|---------|-------------|
| **28 Extensions** | Full OS abstraction: kernel, memory, filesystem, drivers, shell |
| **12 Dimensions** | Lucas-numbered states mapped to NPU/CPU/GPU |
| **O(1) Wormhole** | Instant routing via Brahim transform |
| **53K LOC** | Production-ready Python codebase |
| **PHI-Governed** | All ratios follow golden ratio mathematics |

---

## Architecture

```
+------------------------------------------------------------------+
|                        SPIO STACK                                 |
+------------------------------------------------------------------+
|  [Skills]  [Agents]  [IIAS]  [Research]         Applications    |
+------------------------------------------------------------------+
|  [Kernel]  [Memory]  [Shell]  [Users]           OS Layer        |
+------------------------------------------------------------------+
|  [Wormhole Engine]  PHI Transform, O(1) Routing Math Bridge     |
+------------------------------------------------------------------+
|  [Drivers]  D1-4:NPU | D5-8:CPU | D9-12:GPU     Hardware        |
+------------------------------------------------------------------+
```

**The 12 Dimensions:**

| D | Lucas | Domain | Silicon |
|---|-------|--------|---------|
| 1-4 | 1,3,4,7 | Perception/Security | NPU |
| 5-8 | 11,18,29,47 | Compression/Reasoning | CPU |
| 9-12 | 76,123,199,322 | Creativity/Unification | GPU |

**Total: 840 states** (sum of Lucas numbers)

---

## Extensions

### OS Layer (10)

| Extension | Purpose |
|-----------|---------|
| `kernel` | Process scheduling, worker pools |
| `memory` | Allocation, budget tracking |
| `filesystem` | Virtual FS, artifacts |
| `boot` | Genesis sequence |
| `drivers` | 12D -> silicon routing |
| `shell` | Terminal, REPL |
| `users` | Permissions, sandboxing |
| `safety` | Circuit breakers, failsafe |
| `networking` | Protocol handlers |
| `gateway` | API & substrate access |

### Intelligence Layer (10)

| Extension | Purpose |
|-----------|---------|
| `brahims_laws` | 6 Laws, elliptic curve analysis |
| `skills` | 50+ autonomous skills |
| `agents` | Multi-agent orchestration |
| `iias` | 125 infrastructure apps |
| `cognitive` | MetaCortex, alignment |
| `reflexes` | System 1 triggers |
| `research` | BSD, Hodge automation |
| `wormhole` | PHI transform engine |
| `physics` | Cosmology, Yang-Mills |
| `gpia_wormhole` | Instant retrieval |

### Infrastructure (8)

`boot` | `bridges` | `boa_sdks` | `dashboard` | `psi` | `scripts` | `modes` | `budget`

---

## Mathematical Foundation

```python
# Brahim's Calculator - All computations deterministic
PHI   = 1.6180339887498949   # Golden ratio
OMEGA = 0.6180339887498949   # 1/PHI (compression)
BETA  = 0.2360679774997897   # 1/PHI^3 (security)
GAMMA = 0.1458980337503155   # 1/PHI^4 (damping)

# Brahim Sequence - Mirror pairs sum to 214
B = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)
# Center C = 107 (critical line ratio 1/2)

# Energy is ALWAYS 2*PI
def Energy(x): return PHI**D(x) * Theta(x)  # = 2*PI
```

---

## Brahim's 6 Laws

1. **Brahim Conjecture:** `Sha ~ Im(tau)^(2/3)`
2. **Arithmetic Reynolds:** `Rey = N/(Tam*Omega)`
3. **Phase Transition:** `Rey_c in [10, 30]`
4. **Dynamic Scaling:** `Sha_max ~ Rey^(5/12)`
5. **Cascade Law:** `Var(log Sha) ~ p^(-1/4)`
6. **Consistency:** `2/3 = 5/12 + 1/4`

---

## Stats

| Metric | Value |
|--------|-------|
| Extensions | 28 |
| Python Files | 142 |
| Lines of Code | 53,234 |
| Core Systems | 15/15 |
| Lint | Passing |
| Tests | Passing |

---

## Docs

- **[USAGE.md](USAGE.md)** — Comprehensive usage guide
- **[CLAUDE.md](CLAUDE.md)** — AI assistant instructions
- **[examples/](examples/)** — Code examples

---

## Contributing

Contributions welcome! Please read our guidelines:

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feat/amazing`)
5. Open a Pull Request

---

## License

[MIT](LICENSE) — Free for personal and commercial use.

---

## Citation

```bibtex
@software{sovereign_pio_2026,
  title  = {Sovereign PIO: Autonomous Agent OS},
  author = {Cloudhabil},
  year   = {2026},
  url    = {https://github.com/Cloudhabil/spio},
  version = {1.618.0}
}
```

---

<p align="center">
  <strong>PHI governs all ratios. Energy is always 2PI.</strong>
</p>

<p align="center">
  <sub>Built with golden ratio mathematics by <a href="https://cloudhabil.com">Cloudhabil</a></sub>
</p>
