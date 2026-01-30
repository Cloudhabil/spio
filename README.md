# Sovereign PIO

**Personal Intelligent Operator -- Autonomous Agent Operating System**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Cloudhabil/spio/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version 1.618.1](https://img.shields.io/badge/version-1.618.1-gold.svg)](https://github.com/Cloudhabil/spio)
[![Ruff](https://img.shields.io/badge/linter-ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/Cloudhabil/spio)

Four autonomous layers -- **PIO**, **GPIA**, **ASIOS**, **Moltbot** -- wired into a single runtime. One command boots the full stack: session management, semantic memory, LLM reasoning, resource governance, safety auditing, and multi-channel I/O.

```
spio run                     # terminal + echo mode
spio run --llm ollama        # terminal + Ollama LLM
spio status                  # JSON status of all 4 layers
```

All routing decisions are O(1), all energy conserved at 2&pi;, all capacity governed by 840 discrete Lucas states across 12 cognitive dimensions.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Runtime Architecture](#runtime-architecture)
- [Message Pipeline](#message-pipeline)
- [CLI Reference](#cli-reference)
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
spio --version
# Sovereign PIO v1.618.1

python -c "from sovereign_pio.constants import PHI; print(f'PHI = {PHI}')"
# PHI = 1.6180339887498949
```

---

## Quick Start

### One Command

```bash
spio run
```

This boots all four layers with the terminal channel in echo mode (no external dependencies). Type a message, get a response, all middleware runs.

### With an LLM

```bash
spio run --llm ollama              # local Ollama
spio run --llm ollama --model mistral
spio run --llm openai --api-key sk-...
```

### From Python

```python
import asyncio
from sovereign_pio.runtime import SovereignRuntime, RuntimeConfig

config = RuntimeConfig(llm_provider="echo")
runtime = SovereignRuntime(config)
runtime.boot()

# Process a single message
response = asyncio.run(
    runtime.pio.process("session-1", "What is the golden ratio?")
)
print(response)
# [PIO] Received (query): What is the golden ratio?

# Inspect all layers
print(runtime.status())
# {'booted': True, 'pio': {...}, 'gpia': {...}, 'asios': {...}, 'moltbot': {...}, ...}
```

### IIAS Routing

```python
from extensions.iias import DimensionRouter

router = DimensionRouter()
result = router.route(100.0)  # 100 MB workload -> NPU/CPU/GPU split
print(result["total_time_ms"])
```

---

## Runtime Architecture

`SovereignRuntime` is the central bootstrap that wires the four layers.

```
┌─────────────────────────────────────────────────────────┐
│                    MOLTBOT (Gateway)                     │
│  Terminal  |  Telegram  |  Discord  |  Webhook           │
└──────────────────────┬──────────────────────────────────┘
                       │ on_message
┌──────────────────────▼──────────────────────────────────┐
│                     PIO (Operator)                        │
│  Sessions  |  Intent Detection  |  Middleware Pipeline    │
│                                                           │
│  ┌─ middleware[0]: ASIOS safety    (Governor.check_health)│
│  ├─ middleware[1]: Wavelength audit (WavelengthGate)      │
│  ├─ middleware[2]: Logging                                │
│  ├─ middleware[3]: Memory store                           │
│  └─ default: Memory.search → ReasoningEngine.reason       │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│     GPIA     │ │  ASIOS   │ │  Wavelength  │
│ Memory       │ │ Governor │ │  Gate        │
│ Reasoning    │ │ Gov't    │ │  (12 phases) │
│ Embeddings   │ │ PassBrkr │ │              │
│ (Ollama/OAI) │ │ FailSafe │ │              │
└──────────────┘ └──────────┘ └──────────────┘
```

### The Four Layers

| Layer | Package | Role |
|-------|---------|------|
| **PIO** | `src/pio/` | User-facing operator: sessions, intent detection, middleware pipeline |
| **GPIA** | `src/gpia/` | Intelligence: semantic memory (SimpleEmbedder/Ollama/OpenAI), LLM reasoning engine, dense state |
| **ASIOS** | `src/asios/` | OS runtime: Governor (hardware monitoring), Government (minister routing), PassBroker (agent-to-agent PASS protocol), FailSafe (circuit breaker) |
| **Moltbot** | `src/moltbot/` | Multi-channel gateway: Terminal, Telegram, Discord, Webhook channels |

### Boot Sequence

`SovereignRuntime.boot()` executes in order:

1. **WavelengthGate** -- 12-phase active inference pipeline (sense, correct, persist)
2. **Memory** -- semantic store with SimpleEmbedder fallback
3. **ReasoningEngine** -- optional, Ollama or OpenAI backend
4. **ASIOSRuntime** -- Governor + Government + PassBroker + FailSafe
5. **PIOOperator** -- wired to Memory and ReasoningEngine, 4 middleware attached
6. **Gateway** -- channel registered, handler wired to `pio.process()`
7. **PassBroker providers** -- KNOWLEDGE (from Memory) and CAPABILITY (from Government)

---

## Message Pipeline

Every message flows through the full stack:

```
User input
  │
  ▼
Gateway.on_message(handler)
  │
  ▼
PIOOperator.process(session_id, text)
  │
  ├─► middleware[0]: ASIOS safety check
  │     Governor.check_health() → session.context["asios_health"]
  │     If critical → "[ASIOS] Critical stop active" (short-circuit)
  │
  ├─► middleware[1]: Wavelength audit
  │     WavelengthGate.evaluate(text) → session.context["wavelength"]
  │     Records: density, resonance, safe, converged
  │
  ├─► middleware[2]: Logging
  │     Prints [LOG] Session <id>: <text>
  │
  ├─► middleware[3]: Memory store
  │     Stores substantial messages (>50 chars) to Memory
  │
  └─► default processing:
        Memory.search(query, top_k=3)    → context
        ReasoningEngine.reason(query)    → response
        (or echo fallback: "[PIO] Received (intent): text")
  │
  ▼
Response returned to Gateway → Channel → User
```

---

## CLI Reference

```
spio [command] [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `spio run` | Boot the full runtime and start the gateway |
| `spio status` | Boot in echo mode, print JSON status of all layers |

### Run Options

| Flag | Description |
|------|-------------|
| `--llm {echo,ollama,openai}` | LLM provider (default: echo) |
| `--model NAME` | Model name (default: llama3.2 for Ollama, gpt-4o-mini for OpenAI) |
| `--host URL` | Ollama host URL (default: http://localhost:11434) |
| `--api-key KEY` | OpenAI API key |
| `--telegram TOKEN` | Use Telegram channel with bot token |
| `--discord TOKEN` | Use Discord channel with bot token |

### Legacy Flags

| Flag | Description |
|------|-------------|
| `--version`, `-v` | Print version |
| `--constants`, `-c` | Print Brahim's Calculator constants |
| `--info`, `-i` | Print architecture layer summary |

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
print(f"Total: {stats['total']}")       # 125
print(f"Categories: {len(stats['categories'])}")  # 13
```

---

## Configuration

### pyproject.toml

```toml
[project]
name = "sovereign-pio"
version = "1.618.0"
requires-python = ">=3.10"

[project.scripts]
spio = "sovereign_pio.cli:main"
sovereign-pio = "sovereign_pio.cli:main"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### RuntimeConfig

```python
from sovereign_pio.runtime import RuntimeConfig

config = RuntimeConfig(
    llm_provider="ollama",           # "echo" | "ollama" | "openai"
    llm_model="llama3.2",           # model name
    llm_host="http://localhost:11434",
    openai_api_key="",
    channel="terminal",             # "terminal" | "telegram" | "discord"
    channel_token="",               # bot token for telegram/discord
    wavelength_threshold=0.1,
    memory_persist_path=None,       # path for persistent memory
)
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
| Python files       | 150+      |
| Extensions         | 28        |
| IIAS applications  | 125       |
| Cognitive dims     | 12        |
| Total Lucas states | 840       |
| Goldbach extensions| 6         |
| Integration tests  | 9         |
| Lint (ruff)        | Passing   |
| Type check (mypy)  | Passing   |

---

## Documentation

| Document                                  | Description                            |
|-------------------------------------------|----------------------------------------|
| [USAGE.md](USAGE.md)                      | Comprehensive usage guide              |
| [CLAUDE.md](CLAUDE.md)                    | AI assistant integration instructions  |
| [examples/](examples/)                    | Code examples and notebooks            |

### Examples

| File | Description |
|------|-------------|
| [`examples/main.py`](examples/main.py) | Full stack demo with terminal channel |
| [`examples/telegram_bot.py`](examples/telegram_bot.py) | Telegram bot integration |
| [`examples/discord_bot.py`](examples/discord_bot.py) | Discord bot integration |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/description`
3. Run lint and type checks:
   ```bash
   ruff check .
   mypy src/ extensions/ --ignore-missing-imports
   ```
4. Run tests:
   ```bash
   pytest tests/ -v
   ```
5. Commit with a conventional message: `feat(scope): description`
6. Push and open a Pull Request

All contributions must pass `ruff`, `mypy`, and `pytest` before merge.

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
  version = {1.618.1},
  license = {MIT}
}
```
