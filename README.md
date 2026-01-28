# Sovereign PIO

**A Unified Autonomous Agent Architecture with Mathematical Bridge to Hardware**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Version](https://img.shields.io/badge/Version-1.618.0-gold.svg)](pyproject.toml)
[![Website](https://img.shields.io/badge/Website-cloudhabil.com-blue.svg)](http://cloudhabil.com/)

## Overview

Sovereign PIO (Personal Intelligent Operator) is an autonomous agent architecture that bridges high-level AI operations with hardware through mathematical transformations based on the golden ratio (φ). The system implements a complete OS-like abstraction layer with 20 extensions providing kernel, memory, filesystem, drivers, and wormhole-based routing capabilities.

**Live Site**: [http://cloudhabil.com/](http://cloudhabil.com/)

## Key Statistics

| Metric | Value |
|--------|-------|
| Extensions | 20 |
| Components | 51+ |
| Lines of Code | 18,862+ |
| Dimensions | 12 |
| Total States | 840 |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SPIO Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ Skills  │  │ Agents  │  │Research │  │  IIAS   │   Applications │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                │
│       │            │            │            │                       │
│  ┌────┴────────────┴────────────┴────────────┴────┐                 │
│  │                    Kernel                       │   OS Layer      │
│  │  (Scheduler, Processes, WorkerPool)            │                 │
│  └────────────────────┬───────────────────────────┘                 │
│                       │                                              │
│  ┌────────┬───────────┼───────────┬────────┐                        │
│  │ Memory │  Shell    │  Users    │  FS    │        Services        │
│  └────┬───┘  └────┬───┘  └────┬───┘  └──┬──┘                        │
│       │           │           │          │                           │
│  ┌────┴───────────┴───────────┴──────────┴────┐                     │
│  │              Wormhole Engine                │   Math Bridge      │
│  │  (PHI transform, O(1) routing)             │                     │
│  └────────────────────┬───────────────────────┘                     │
│                       │                                              │
│  ┌────────────────────┴───────────────────────┐                     │
│  │              Drivers                        │   Hardware         │
│  │  D1-4: NPU  │  D5-8: CPU  │  D9-12: GPU   │   Abstraction      │
│  └────────────────────┬───────────────────────┘                     │
│                       │                                              │
│  ┌────────────────────┴───────────────────────┐                     │
│  │         Physical Hardware                   │   Silicon          │
│  │         NPU | CPU | GPU | RAM | SSD        │                     │
│  └────────────────────────────────────────────┘                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Extensions

### OS Layer (8 extensions)

| Extension | Description | Key Components |
|-----------|-------------|----------------|
| **kernel** | Process scheduling and orchestration | `KernelSubstrate`, `AdaptiveScheduler`, `ProcessManager` |
| **memory** | Memory allocation and budget tracking | `MemoryPool`, `VirtualMemory`, `DenseStateMemory`, `BudgetLedger` |
| **filesystem** | Virtual filesystem and artifact management | `VirtualFS`, `ArtifactClassifier`, `FilesystemGardener` |
| **boot** | Genesis sequence and service management | `Boot`, `Genesis`, `BootLoader`, `ServiceManager` |
| **drivers** | 12-dimension to silicon routing | `DimensionRouter`, `NPUDriver`, `GPUDriver`, `CPUDriver` |
| **users** | Unix-style permissions and sandboxing | `Users`, `PermissionManager`, `Sandbox`, `ACL` |
| **shell** | Terminal execution and REPL | `Shell`, `TerminalExecutor`, `CommandParser`, `REPL` |
| **wormhole** | Mathematical bridge to hardware | `BrahimWormholeEngine`, `WormholeRouter`, `WormholeTransform` |

### Application Layer (12 extensions)

| Extension | Description |
|-----------|-------------|
| **skills** | 50+ autonomous skills with PHI-based selection |
| **agents** | Multi-agent orchestration (`AlphaAgent`, `AgentRegistry`) |
| **iias** | Intelligent Infrastructure Agent System |
| **research** | BSD, Erdos-Straus, Hodge research automation |
| **reflexes** | Trigger/action system (`ReflexRunner`) |
| **psi** | Privacy and security (`BrahimRouter`, `FingerprintBlocker`) |
| **dashboard** | Monitoring and control UI |
| **boa_sdks** | Brahim Engine SDK integrations |
| **brahims_laws** | BSD conjecture mathematical framework |
| **gpia_wormhole** | GPIA wormhole topology |
| **core** | Foundation utilities |
| **scripts** | 93 utility scripts catalog |

## Mathematical Foundation

### Brahim's Calculator

All computations are deterministic using these constants:

```python
PHI   = 1.6180339887498949   # Golden ratio
ALPHA = 0.3819660112501051   # 1/PHI^2  (Creation)
BETA  = 0.2360679774997897   # 1/PHI^3  (Security threshold)
GAMMA = 0.1458980337503155   # 1/PHI^4  (Damping)
```

### Brahim Sequence

```python
BRAHIM_SEQUENCE = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)
# Mirror pairs sum to 214: 27+187, 42+172, 60+154, 75+139, 97+117
```

### 12-Dimension Model

| Dimension | Lucas | Domain | Silicon |
|-----------|-------|--------|---------|
| D1 | 1 | Perception | NPU |
| D2 | 3 | Attention | NPU |
| D3 | 4 | Security | NPU |
| D4 | 7 | Stability | NPU |
| D5 | 11 | Compression | CPU |
| D6 | 18 | Harmony | CPU |
| D7 | 29 | Reasoning | CPU |
| D8 | 47 | Prediction | CPU |
| D9 | 76 | Creativity | GPU |
| D10 | 123 | Wisdom | GPU |
| D11 | 199 | Integration | GPU |
| D12 | 322 | Unification | GPU |

**Total States**: 840 (sum of Lucas numbers)

## Installation

```bash
git clone https://github.com/Cloudhabil/spio.git
cd spio
pip install -e .
```

## Quick Start

```python
# Boot the system
from extensions.boot import Boot
boot = Boot()
boot.run()

# Use wormhole compression (O(1))
from extensions.wormhole import BrahimWormholeEngine
engine = BrahimWormholeEngine()
result = engine.transform([1, 2, 3, 4, 5], iterations=3)
print(f"Compression: {result.compression_ratio}")  # 0.236 (BETA)

# Route to hardware
from extensions.drivers import Drivers
drivers = Drivers()
drivers.initialize()
silicon = drivers.route_dimension(7)  # Reasoning -> CPU

# Memory management
from extensions.memory import Memory
memory = Memory(max_mb=512)
block = memory.pool.allocate(1024 * 100, owner="my-task")

# User permissions
from extensions.users import Users, AgentRole
users = Users()
agent = users.create_user("worker", AgentRole.STANDARD)
```

See [USAGE.md](USAGE.md) for comprehensive documentation.

## Wormhole Bridge

The wormhole extension provides the mathematical bridge between SPIO and hardware:

```python
from extensions.wormhole import BrahimWormholeEngine, WormholeRouter

# Create engine
engine = BrahimWormholeEngine(throat_radius=1.0)

# Validate geometry
validation = engine.validate()
# {'geometry_valid': True, 'traversable': True, 'stable': True, ...}

# O(1) compression via wormhole transform
# W(x) = x/PHI + C_bar * ALPHA
result = engine.transform(data, iterations=3)
# Compression ratio converges to BETA (0.236)

# Route packets through wormhole shortcuts
router = WormholeRouter()
router.add_packet("doc1", "Python tutorial")
router.add_packet("doc2", "Python ML library")
router.create_wormhole("doc1", "doc2")  # O(1) shortcut

result = router.route("Python")
# Route type: direct (used wormhole), Latency: 0.01ms
```

## Security Model

- **Command Validation**: Blacklist + heuristic pattern detection
- **Filesystem Sandboxing**: Zone-based access control
- **User Permissions**: Unix-style (ROOT/ADMIN/STANDARD/GUEST)
- **Resource Governance**: VRAM cliff protection, thermal throttling

## Project Structure

```
sovereign-pio/
├── extensions/
│   ├── agents/          # Multi-agent orchestration
│   ├── boa_sdks/        # SDK integrations
│   ├── boot/            # Genesis & boot sequence
│   ├── brahims_laws/    # Mathematical framework
│   ├── core/            # Foundation utilities
│   ├── dashboard/       # Monitoring UI
│   ├── drivers/         # Hardware abstraction
│   ├── filesystem/      # Virtual filesystem
│   ├── gpia_wormhole/   # GPIA integration
│   ├── iias/            # Intelligent agents
│   ├── kernel/          # Process scheduling
│   ├── memory/          # Memory management
│   ├── psi/             # Privacy/security
│   ├── reflexes/        # Trigger/action
│   ├── research/        # Research automation
│   ├── scripts/         # Utility scripts
│   ├── shell/           # Terminal executor
│   ├── skills/          # 50+ skills
│   ├── users/           # Permissions
│   └── wormhole/        # Mathematical bridge
├── index.html           # GitHub Pages landing
├── USAGE.md             # Comprehensive guide
└── README.md            # This file
```

## References

1. Livio, M. (2002). *The Golden Ratio: The Story of PHI*
2. Morris, M. S., & Thorne, K. S. (1988). "Wormholes in spacetime"
3. Lucas, É. (1891). *Théorie des nombres*

## License

MIT License - See [LICENSE](LICENSE)

## Citation

```bibtex
@software{sovereign_pio_2026,
  title = {Sovereign PIO: Autonomous Agent Architecture with Mathematical Bridge},
  author = {Cloudhabil},
  year = {2026},
  url = {https://github.com/Cloudhabil/spio},
  version = {1.618.0}
}
```

---

*Sovereign PIO — Deterministic Intelligence Through Golden Ratio Governance*

**PHI governs all ratios. Energy is always 2π.**
