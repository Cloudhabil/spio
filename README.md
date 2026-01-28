# Sovereign PIO

**A Unified Autonomous Agent Architecture for Multi-Channel Intelligent Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Version](https://img.shields.io/badge/Version-1.618.0-green.svg)](pyproject.toml)

## Abstract

Sovereign PIO (Personal Intelligent Operator) presents a novel architecture for autonomous agent systems, integrating deterministic computation frameworks with modern large language model orchestration. The system implements a four-layer abstraction model comprising interface management (PIO), cognitive reasoning (GPIA), runtime governance (ASIOS), and multi-channel communication (Moltbot). Central to the architecture is Brahim's Calculator, a mathematical framework based on the golden ratio (φ) that provides deterministic computation guarantees across all system operations.

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Mathematical Foundation](#mathematical-foundation)
- [Component Specifications](#component-specifications)
- [Installation](#installation)
- [Usage](#usage)
- [Security Model](#security-model)
- [References](#references)
- [License](#license)

## Introduction

The proliferation of large language models has created demand for robust agent architectures capable of operating across heterogeneous communication channels while maintaining deterministic behavior guarantees. Sovereign PIO addresses this challenge through a layered architecture that separates concerns across four distinct subsystems:

1. **PIO** (Personal Intelligent Operator): User interface and session management
2. **GPIA** (General Purpose Intelligence Architecture): Reasoning and memory systems
3. **ASIOS** (Autonomous System Input/Output Supervisor): Runtime governance and resource management
4. **Moltbot**: Multi-channel messaging gateway

This separation enables independent scaling, testing, and deployment of each subsystem while maintaining coherent system behavior through well-defined interfaces.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SOVEREIGN PIO v1.618.0                           │
│              Personal Intelligent Operator · Autonomous Edition             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│    │       PIO       │◄──►│      GPIA       │◄──►│      ASIOS      │       │
│    │    Interface    │    │   Intelligence  │    │     Runtime     │       │
│    │                 │    │                 │    │                 │       │
│    │ • Sessions      │    │ • Reasoning     │    │ • FailSafe      │       │
│    │ • Intent Det.   │    │ • Memory        │    │ • Governor      │       │
│    │ • Middleware    │    │ • Embeddings    │    │ • Hardware Mon. │       │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘       │
│              │                    │                      │                  │
│              └────────────────────┼──────────────────────┘                  │
│                                   ▼                                         │
│    ┌───────────────────────────────────────────────────────────────────┐   │
│    │                           MOLTBOT                                  │   │
│    │                    Multi-Channel Gateway                           │   │
│    │                                                                    │   │
│    │    Terminal   Webhook   Telegram   Discord   Slack   WhatsApp     │   │
│    └───────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Mathematical Foundation

### Brahim's Calculator

All numerical computations within Sovereign PIO are governed by Brahim's Calculator, a deterministic mathematical framework ensuring reproducible outputs across system operations.

#### Fundamental Constants

| Constant | Symbol | Value | Semantic Role |
|----------|--------|-------|---------------|
| Golden Ratio | φ (PHI) | 1.6180339887498949 | Universal scaling factor |
| Creation | α (ALPHA) | φ | Forward transformation |
| Return | ω (OMEGA) | 1/φ = 0.6180339887498949 | Inverse transformation |
| Security | β (BETA) | 1/φ³ = 0.2360679774997897 | Safety margin coefficient |
| Damping | γ (GAMMA) | 1/φ⁴ = 0.1458980337503155 | Attenuation factor |

#### Core Functions

```python
def D(x: float) -> float:
    """Dimension mapping function. Maps value x to φ-space dimension."""
    return -log(x) / log(φ)

def Θ(x: float) -> float:
    """Phase function. Converts value to angular phase."""
    return 2π × x

def E(x: float) -> float:
    """Energy function. Demonstrates conservation: E(x) = 2π ∀x > 0."""
    return φ^D(x) × Θ(x)  # ≡ 2π

def x_from_D(d: float) -> float:
    """Inverse dimension mapping. Recovers x from dimension d."""
    return 1 / φ^d
```

#### Twelve-Dimension Model

The system organizes cognitive operations across twelve dimensions, each with Lucas number capacity:

| Dimension | L(D) | Domain | Silicon Affinity |
|-----------|------|--------|------------------|
| D₁ | 1 | Perception | NPU |
| D₂ | 3 | Attention | NPU |
| D₃ | 4 | Security | NPU |
| D₄ | 7 | Stability | NPU |
| D₅ | 11 | Compression | CPU |
| D₆ | 18 | Harmony | CPU |
| D₇ | 29 | Reasoning | CPU |
| D₈ | 47 | Prediction | CPU |
| D₉ | 76 | Creativity | GPU |
| D₁₀ | 123 | Wisdom | GPU |
| D₁₁ | 199 | Integration | GPU |
| D₁₂ | 322 | Unification | GPU |

**Total State Capacity**: Σ L(Dᵢ) = 840 discrete states

## Component Specifications

### Implementation Status

| Component | Status | LOC | Description |
|-----------|--------|-----|-------------|
| Calculator | Complete | ~150 | PHI, D(), Θ(), E(), lucas() |
| Security | Complete | ~200 | CommandValidator, FilesystemGuard |
| FailSafe | Complete | ~80 | Circuit breaker pattern |
| Governor | Complete | ~300 | nvidia-smi integration, VRAM/thermal monitoring |
| Memory | Complete | ~350 | Embeddings (Simple/Ollama/OpenAI), vector search |
| Reasoning | Complete | ~400 | LLM client (Ollama/OpenAI), dimension routing |
| PIO Operator | Complete | ~350 | Async sessions, intent detection, middleware |
| Moltbot Channels | Complete | ~800 | Terminal, Webhook, Telegram, Discord |
| Gateway | Complete | ~200 | Multi-channel orchestration |

**Total**: ~2,830 lines of implementation code

### Subsystem Details

#### PIO (Personal Intelligent Operator)

- **Session Management**: Stateful conversation tracking with UUID-based identification
- **Intent Detection**: Pattern-based classification (Query, Command, Creative, Analysis)
- **Middleware Pipeline**: Composable request/response processing chain
- **Async Processing**: Full asyncio support for concurrent operations

#### GPIA (Intelligence Layer)

- **Memory System**: Semantic storage with embedding-based retrieval
  - SimpleEmbedder: Hash-based deterministic vectors (fallback)
  - OllamaEmbedder: Local embedding via Ollama API
  - OpenAIEmbedder: Cloud embedding via OpenAI API
- **Reasoning Engine**: LLM orchestration with dimension-aware routing
  - Ollama backend support
  - OpenAI backend support
  - Streaming response capability
- **Multi-Model Orchestrator**: Dimension-to-model mapping for specialized processing

#### ASIOS (Runtime Layer)

- **FailSafe**: Circuit breaker implementation
  - Sliding window failure tracking
  - Configurable threshold and cooldown
  - Thread-safe operation
- **Governor**: Hardware resource management
  - Real-time nvidia-smi integration
  - VRAM utilization monitoring (81.25% cliff protection)
  - Thermal throttling (78°C threshold)
  - Disk space monitoring

#### Moltbot (Communication Layer)

- **Gateway**: Centralized channel management with auto-reconnection
- **Channels**:
  - Terminal: Local CLI interaction
  - Webhook: HTTP-based integration with HMAC authentication
  - Telegram: Full Bot API implementation with long polling
  - Discord: Gateway WebSocket with HTTP API

## Installation

### Requirements

- Python 3.10 or higher
- Optional: NVIDIA GPU with nvidia-smi for hardware monitoring
- Optional: Ollama for local LLM inference

### Standard Installation

```bash
git clone https://github.com/Cloudhabil/spio.git
cd spio
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

## Usage

### Quick Start

```bash
# Run component demonstration
python examples/main.py --demo

# Interactive terminal session
python examples/main.py
```

### Programmatic Usage

```python
import asyncio
from sovereign_pio import PHI, D, Energy
from pio import PIOOperator
from gpia import Memory, ReasoningEngine, ModelConfig
from moltbot import Gateway, TerminalChannel

async def main():
    # Initialize components
    memory = Memory()
    reasoning = ReasoningEngine(ModelConfig.ollama())

    pio = PIOOperator()
    pio.set_memory(memory)
    pio.set_reasoning_engine(reasoning)

    # Process a query
    response = await pio.process(
        session_id="demo",
        user_input="Explain the golden ratio."
    )
    print(response)

asyncio.run(main())
```

### Channel Integration

#### Telegram Bot

```bash
export TELEGRAM_BOT_TOKEN="your-token"
python examples/telegram_bot.py
```

#### Discord Bot

```bash
export DISCORD_BOT_TOKEN="your-token"
python examples/discord_bot.py
```

## Security Model

Sovereign PIO implements defense-in-depth security:

### Command Validation

- Binary blacklist (shutdown, reboot, mkfs, dd, nc, shell interpreters)
- Heuristic pattern detection (path traversal, command substitution)
- Lexical analysis with safe parsing

### Filesystem Sandboxing

- Zone-based access control (read zones, write zones)
- Path traversal prevention
- Project root containment

### Authentication

- Multi-tier authentication (local, network, token)
- HMAC signature verification for webhooks
- Timing-safe credential comparison

### Resource Governance

- VRAM cliff protection (prevents Windows DWM instability)
- Thermal throttling
- Circuit breaker isolation

## Project Statistics

```
Repository:     github.com/Cloudhabil/spio
Version:        1.618.0
License:        MIT
Language:       Python 3.10+
Total Files:    58
Lines of Code:  4,089
Test Coverage:  Core mathematical functions
```

## References

1. Livio, M. (2002). *The Golden Ratio: The Story of PHI, the World's Most Astonishing Number*. Broadway Books.
2. Lucas, É. (1891). *Théorie des nombres*. Gauthier-Villars.
3. Nygard, M. T. (2007). *Release It!: Design and Deploy Production-Ready Software*. Pragmatic Bookshelf. (Circuit breaker pattern)
4. Vaswani, A., et al. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems*.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{sovereign_pio_2026,
  title = {Sovereign PIO: A Unified Autonomous Agent Architecture},
  author = {Cloudhabil},
  year = {2026},
  url = {https://github.com/Cloudhabil/spio},
  version = {1.618.0}
}
```

---

*Sovereign PIO — Deterministic Intelligence Through Golden Ratio Governance*
