# SPIO Usage Guide

## Quick Start

```python
# Import the main extensions
from extensions.boot import Boot, Genesis
from extensions.kernel import Kernel, AdaptiveScheduler
from extensions.drivers import Drivers, DimensionRouter
from extensions.wormhole import BrahimWormholeEngine, WormholeRouter
from extensions.memory import Memory, BudgetLedger
from extensions.shell import Shell
from extensions.users import Users, AgentRole
from extensions.skills import AutonomousSkillSelector, ALL_SKILLS
```

## 1. Boot Sequence

```python
from extensions.boot import Boot, BootConfig

# Configure boot
config = BootConfig(
    mode="sovereign",
    skip_preflight=False,
    timeout_ms=30000
)

# Boot the system
boot = Boot(config)
success = boot.run()

print(f"Boot status: {boot.get_status()}")
# Output: {'booted': True, 'genesis': {...}, 'services': {...}}
```

## 2. Genesis (PIO Birth)

```python
from extensions.boot import Genesis, GenesisPhase

# Create genesis
genesis = Genesis()
genesis.begin()

# Step through phases
while genesis.function.phase != GenesisPhase.PIO_OPERATIONAL:
    phase = genesis.step(0.1)
    print(f"Phase: {phase.value}, Dimensions: {genesis.function._dimensions_emerged}")

# Complete genesis
genesis.complete()
print(genesis.get_state())
# Output: {'t': 1.0, 'phase': 'pio_operational', 'dimensions_emerged': 12, ...}
```

## 3. Wormhole Transform (The Mathematical Bridge)

```python
from extensions.wormhole import BrahimWormholeEngine, quick_transform, PHI, BETA

# Create engine
engine = BrahimWormholeEngine(throat_radius=1.0)

# Validate wormhole geometry
validation = engine.validate()
print(f"All valid: {validation['all_valid']}")  # True

# Apply wormhole transform (O(1) compression)
data = [100, 50, 200, 75, 150, 80, 120, 90, 160, 110]
result = engine.transform(data, iterations=3)

print(f"Compression ratio: {result.compression_ratio:.4f}")  # ~0.236 (BETA)
print(f"Output: {result.output_vector[:5]}")

# Quick transform without engine
compressed = quick_transform([1, 2, 3, 4, 5], iterations=5)
```

## 4. Dimension Routing (12D → Silicon)

```python
from extensions.drivers import Drivers, DimensionRouter, SiliconLayer

# Initialize drivers
drivers = Drivers()
drivers.initialize()

# Route dimensions to hardware
for dim in range(1, 13):
    silicon = drivers.route_dimension(dim)
    print(f"D{dim} → {silicon.value}")

# Output:
# D1 → npu   (Perception)
# D2 → npu   (Attention)
# D3 → npu   (Security)
# D4 → npu   (Stability)
# D5 → cpu   (Compression)
# D6 → cpu   (Harmony)
# D7 → cpu   (Reasoning)
# D8 → cpu   (Prediction)
# D9 → gpu   (Creativity)
# D10 → gpu  (Wisdom)
# D11 → gpu  (Integration)
# D12 → gpu  (Unification)

# Route by task type
silicon = drivers.route_task("reasoning")  # → CPU
silicon = drivers.route_task("creativity")  # → GPU
```

## 5. Wormhole Router (Information Retrieval)

```python
from extensions.wormhole import WormholeRouter, RouteType

# Create router
router = WormholeRouter()

# Add data packets
router.add_packet("doc1", "Python programming tutorial", embedding=[0.1]*10)
router.add_packet("doc2", "Machine learning basics", embedding=[0.2]*10)
router.add_packet("doc3", "Python ML library usage", embedding=[0.15]*10)

# Create wormhole shortcuts
router.create_wormhole("doc1", "doc3")  # Connect related docs

# Route a query
result = router.route("Python tutorial", k=3)

print(f"Route type: {result.route_type.value}")  # 'direct' (used wormhole)
print(f"Wormholes used: {result.wormholes_used}")
print(f"Results: {len(result.packets)}")
print(f"Stats: {router.get_stats()}")
```

## 6. Kernel & Process Scheduling

```python
from extensions.kernel import Kernel, ProcessPriority

# Get kernel
kernel = Kernel()

# Create processes
pid1 = kernel.create_process("worker-1", ProcessPriority.NORMAL)
pid2 = kernel.create_process("worker-2", ProcessPriority.HIGH)

# Schedule work
kernel.schedule_work(lambda: print("Task executed"))

# Get kernel stats
print(kernel.get_stats())
```

## 7. Memory Management

```python
from extensions.memory import Memory, BudgetLedger

# Create memory system
memory = Memory(max_mb=1024)

# Allocate memory block
block = memory.pool.allocate(1024 * 1024, owner="my-task")  # 1MB
print(f"Allocated: {block.id}, {block.size_bytes} bytes")

# Token budget management
budget = memory.budget
entry = budget.allocate(
    task_id="task-1",
    agent="gpt-4",
    model="gpt-4",
    tokens=10000
)
budget.use(entry.id, 500)  # Use 500 tokens
print(budget.get_stats())

# Dense state memory (vector storage)
memory.dense.store("Context about AI", embedding=[0.1]*384, source="doc1")
context, metadata = memory.dense.retrieve([0.1]*384, max_tokens=2000)
```

## 8. Shell & Command Execution

```python
from extensions.shell import Shell, CommandStatus

# Create shell
shell = Shell()

# Run commands
result = shell.run("python --version", timeout_s=10)
print(f"Status: {result.status.name}")
print(f"Output: {result.stdout}")

# Change directory
shell.cd("/tmp")

# Set/get variables
shell.set_var("MY_VAR", "value")
print(shell.get_var("MY_VAR"))
```

## 9. User Permissions & Sandbox

```python
from extensions.users import Users, AgentRole, SandboxConfig, Sandbox

# Create users system
users = Users()

# Create a new user
agent = users.create_user("my-agent", AgentRole.STANDARD)
print(f"User: {agent.username}, UID: {agent.uid}, Role: {agent.role.name}")

# Check authorization
from pathlib import Path
authorized = users.authorize("my-agent", "read", Path("/data/file.txt"))

# Create sandbox
config = SandboxConfig(
    allowed_paths=[Path("/safe/path")],
    blocked_commands=["rm -rf"],
    network_allowed=False
)
sandbox = users.create_sandbox(config)

# Check if command is allowed
if sandbox.check_command("python script.py"):
    print("Command allowed")
```

## 10. Skills & Autonomous Selection

```python
from extensions.skills import AutonomousSkillSelector, ALL_SKILLS

# Create skill selector
selector = AutonomousSkillSelector()

# Get available skills
print(f"Available skills: {len(ALL_SKILLS)}")

# Select skill based on context
context = {"task": "analyze code", "language": "python"}
selected = selector.select(context)
print(f"Selected: {selected}")

# Execute with autonomous selection
result = selector.select_and_execute(
    context={"task": "summarize text"},
    input_data="Long text to summarize..."
)
```

## Full Integration Example

```python
from extensions.boot import Boot, BootConfig
from extensions.kernel import Kernel
from extensions.drivers import Drivers
from extensions.wormhole import BrahimWormholeEngine
from extensions.memory import Memory
from extensions.shell import Shell
from extensions.users import Users, AgentRole

# 1. Boot system
boot = Boot(BootConfig(mode="sovereign"))
boot.run()

# 2. Initialize subsystems
kernel = Kernel()
drivers = Drivers()
drivers.initialize()
memory = Memory(max_mb=512)
shell = Shell()
users = Users()

# 3. Create wormhole engine for compression
engine = BrahimWormholeEngine()

# 4. Create agent user
agent = users.create_user("worker-agent", AgentRole.STANDARD)

# 5. Allocate resources
budget_entry = memory.budget.allocate("task-1", "worker-agent", "local", 50000)

# 6. Route task to silicon
silicon = drivers.route_task("reasoning")  # → CPU
print(f"Task routed to: {silicon.value}")

# 7. Execute work
result = shell.run("echo 'Hello from SPIO'")
print(result.stdout)

# 8. Apply wormhole compression to results
compressed = engine.transform([ord(c) for c in result.stdout[:10]])
print(f"Compression: {compressed.compression_ratio:.2%}")

# 9. Cleanup
memory.budget.release(budget_entry.id)
boot.shutdown()
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SPIO Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ Skills  │  │ Agents  │  │Research │  │Dashboard│   Applications │
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
│  │  D1-4→NPU  D5-8→CPU  D9-12→GPU            │   Abstraction      │
│  └────────────────────┬───────────────────────┘                     │
│                       │                                              │
│  ┌────────────────────┴───────────────────────┐                     │
│  │         Physical Hardware                   │   Silicon          │
│  │         NPU | CPU | GPU | RAM | SSD        │                     │
│  └────────────────────────────────────────────┘                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Constants (Brahim's Calculator)

```python
PHI   = 1.618033988749895   # Golden ratio
ALPHA = 0.381966011250105   # 1/PHI^2
BETA  = 0.236067977499790   # 1/PHI^3 (Security threshold)
GAMMA = 0.145898033750315   # 1/PHI^4 (Damping)

# Brahim Sequence (mirror pairs sum to 214)
BRAHIM_SEQUENCE = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)

# Lucas numbers for 12 dimensions
LUCAS = [1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199, 322]  # 840 total states
```
