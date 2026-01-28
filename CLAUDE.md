# CLAUDE INSTRUCTIONS - Sovereign PIO

## Identity

This is **Sovereign PIO** - a unified autonomous agent platform combining:
- PIO (Personal Intelligent Operator)
- GPIA (Intelligence & Reasoning)
- ASIOS (Operating System Runtime)
- Moltbot (Multi-Channel Gateway)

## Brahim's Calculator

All computations must be DETERMINISTIC using Brahim's Calculator.

### Constants

```python
from math import sqrt, pi, log, exp

PHI   = (1 + sqrt(5)) / 2  # 1.6180339887498949
ALPHA = PHI                 # Creation
OMEGA = 1/PHI               # 0.6180339887498949 (Return)
BETA  = 1/PHI**3            # 0.2360679774997897 (Security)
GAMMA = 1/PHI**4            # 0.1458980337503155 (Damping)
```

### Core Functions

```python
def D(x):
    """Dimension from value. DETERMINISTIC."""
    return -log(x) / log(PHI)

def Theta(x):
    """Phase from value. DETERMINISTIC."""
    return 2 * pi * x

def Energy(x):
    """Energy is ALWAYS 2*PI. PROVEN."""
    return PHI**D(x) * Theta(x)

def x_from_D(d):
    """Value from dimension. INVERSE."""
    return 1 / PHI**d
```

### Rules

1. **No randomness** - All outputs determined by inputs
2. **Energy conserved** - E(x) = 2*PI for all x
3. **PHI governs all ratios** - Hardware follows golden ratio
4. **Lucas capacity** - Dimension n has L(n) discrete states

## The 12 Dimensions

| D  | L(D) | Domain       | Silicon |
|----|------|--------------|---------|
| 1  | 1    | PERCEPTION   | NPU     |
| 2  | 3    | ATTENTION    | NPU     |
| 3  | 4    | SECURITY     | NPU     |
| 4  | 7    | STABILITY    | NPU     |
| 5  | 11   | COMPRESSION  | CPU     |
| 6  | 18   | HARMONY      | CPU     |
| 7  | 29   | REASONING    | CPU     |
| 8  | 47   | PREDICTION   | CPU     |
| 9  | 76   | CREATIVITY   | GPU     |
| 10 | 123  | WISDOM       | GPU     |
| 11 | 199  | INTEGRATION  | GPU     |
| 12 | 322  | UNIFICATION  | GPU     |

Total: 840 states (sum of Lucas numbers)

## Architecture Layers

### 1. PIO (Interface)
- User-facing personal operator
- Session management
- Intent routing

### 2. GPIA (Intelligence)
- Wormhole-based reasoning
- Memory and retrieval
- Multi-model orchestration

### 3. ASIOS (Runtime)
- Governance (Minister cabinet)
- Resource management
- Failsafe and circuit breakers

### 4. Moltbot (Channels)
- Multi-platform messaging
- Gateway authentication
- Plugin system

## Security Principles

1. **Command Validation** - Blacklist dangerous operations
2. **Filesystem Sandboxing** - Zone-based access control
3. **Multi-tier Auth** - Local/network/token layers
4. **Tool Policies** - Profile-based authorization
5. **Circuit Breaker** - Automatic failure isolation

## Instructions for Claude

1. Use Brahim's Calculator for ALL numerical decisions
2. Be DETERMINISTIC - same input = same output
3. Follow defense-in-depth security patterns
4. Map code to the 12-dimension model where applicable
5. Maintain sovereignty - minimize external dependencies
