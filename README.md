# Sovereign PIO

**Personal Intelligent Operator - Autonomous Edition**

Sovereign PIO is a unified autonomous agent platform combining:

- **PIO** - Personal Intelligent Operator (interface layer)
- **GPIA** - Intelligence and reasoning engine
- **ASIOS** - Autonomous operating system runtime
- **Moltbot** - Multi-channel messaging gateway

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SOVEREIGN PIO                               │
│         Personal Intelligent Operator · Autonomous Edition       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │    PIO      │ ←→ │    GPIA     │ ←→ │   ASIOS     │          │
│  │  Interface  │    │ Intelligence│    │   Runtime   │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│         ↑                  ↑                  ↑                  │
│         └──────────────────┼──────────────────┘                  │
│                            ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      MOLTBOT                                 ││
│  │  WhatsApp · Telegram · Discord · Signal · Slack · iMessage  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Principles

| Principle | Description |
|-----------|-------------|
| **Sovereign** | Self-governing, autonomous, no external dependencies for core operation |
| **Deterministic** | All computations follow Brahim's Calculator (PHI-based) |
| **Secure** | Defense in depth across all layers |
| **Multi-Channel** | Communicate through any platform |

## Brahim's Calculator Constants

All computations are deterministic using these constants:

```python
PHI   = (1 + sqrt(5)) / 2  # 1.6180339887498949
ALPHA = PHI                 # Creation
OMEGA = 1/PHI               # 0.6180339887498949 (Return)
BETA  = 1/PHI**3            # 0.2360679774997897 (Security)
GAMMA = 1/PHI**4            # 0.1458980337503155 (Damping)
```

## Project Structure

```
sovereign-pio/
├── src/
│   ├── pio/          # Personal Intelligent Operator
│   ├── gpia/         # Intelligence & Reasoning
│   ├── asios/        # Operating System Runtime
│   ├── moltbot/      # Multi-Channel Gateway
│   ├── security/     # Security Controls
│   └── core/         # Shared Core Utilities
├── config/           # Configuration Files
├── scripts/          # Utility Scripts
├── tests/            # Test Suite
└── docs/             # Documentation
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sovereign-pio.git
cd sovereign-pio

# Install dependencies
pip install -e .
npm install  # For Moltbot TypeScript components

# Run Sovereign PIO
python -m sovereign_pio
```

## Security

Sovereign PIO implements defense-in-depth security:

- **Command Validation** - Blacklist + heuristic analysis
- **Filesystem Sandboxing** - Zone-based read/write control
- **Multi-tier Authentication** - Local/network/token layers
- **Tool Policies** - Profile-based authorization
- **Circuit Breaker** - Automatic failure isolation
- **Audit Trail** - Cryptographic verification

## License

See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.
