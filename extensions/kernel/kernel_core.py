"""
Kernel Core - Process Scheduler, Substrate, and Mode Management

Level 6 ASI Architecture with 22 tiers of system components.
PHI-governed resource allocation and scheduling.

Based on: CLI-main/src/core/kernel/*
"""

import hashlib
import heapq
import logging
import math
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

# Brahim's Calculator constants
PHI = (1 + math.sqrt(5)) / 2
BETA = 1 / (PHI ** 3)
GAMMA = 1 / (PHI ** 4)
OMEGA = 1 / PHI

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TaskPriority(Enum):
    """Task priority levels (lower value = higher priority)."""
    CRITICAL = 1   # Must run immediately
    HIGH = 2       # Important
    MEDIUM = 3     # Standard
    LOW = 4        # Background
    IDLE = 5       # Only when nothing else


class TaskState(Enum):
    """Task lifecycle states."""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class WorkerState(Enum):
    """Worker thread/process states."""
    IDLE = auto()
    BUSY = auto()
    BLOCKED = auto()
    TERMINATED = auto()


class ModeType(Enum):
    """Operational mode types."""
    SOVEREIGN = "sovereign"
    TEACHING = "teaching"
    GARDENER = "gardener"
    FORENSIC = "forensic"
    MAINTENANCE = "maintenance"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Task:
    """A schedulable task unit."""
    id: str
    name: str
    priority: TaskPriority = TaskPriority.MEDIUM
    state: TaskState = TaskState.PENDING
    func: Optional[Callable] = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout_ms: int = 30000
    retries: int = 0
    max_retries: int = 3
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "Task") -> bool:
        """For heap comparison (lower priority value = higher priority)."""
        return self.priority.value < other.priority.value


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    retries_used: int = 0


@dataclass
class HardwareSnapshot:
    """Hardware state measurement."""
    vram_used_mb: float = 0
    vram_total_mb: float = 0
    vram_free_mb: float = 0
    ram_used_mb: float = 0
    ram_total_mb: float = 0
    ram_free_mb: float = 0
    cpu_percent: float = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def vram_util_percent(self) -> float:
        if self.vram_total_mb == 0:
            return 0
        return (self.vram_used_mb / self.vram_total_mb) * 100

    @property
    def can_fit_model(self, size_mb: float = 4096) -> bool:
        """Check if model fits with 1GB reserve."""
        return self.vram_free_mb >= (size_mb + 1024)


@dataclass
class ResourceProfile:
    """Resource consumption profile for a task/agent."""
    name: str
    vram_used_mb: float = 0
    time_seconds: float = 0
    tokens_generated: int = 0
    tokens_per_second: float = 0
    success: bool = True
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModeTransition:
    """Mode transition request."""
    from_mode: str
    to_mode: str
    reason: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TASK QUEUE
# =============================================================================

class TaskQueue:
    """
    Priority-based task queue with preemption support.

    Uses a min-heap for O(log n) insertion and extraction.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._heap: List[Tuple[int, float, Task]] = []
        self._lock = threading.Lock()
        self._task_map: Dict[str, Task] = {}
        self._counter = 0

    def push(self, task: Task) -> bool:
        """Add task to queue. Returns False if full."""
        with self._lock:
            if len(self._heap) >= self.max_size:
                return False
            # Heap entries: (priority, insertion_order, task)
            entry = (task.priority.value, self._counter, task)
            self._counter += 1
            heapq.heappush(self._heap, entry)
            self._task_map[task.id] = task
            task.state = TaskState.QUEUED
            return True

    def pop(self) -> Optional[Task]:
        """Get highest priority task."""
        with self._lock:
            while self._heap:
                _, _, task = heapq.heappop(self._heap)
                if task.id in self._task_map:
                    del self._task_map[task.id]
                    return task
            return None

    def peek(self) -> Optional[Task]:
        """View highest priority task without removing."""
        with self._lock:
            if self._heap:
                return self._heap[0][2]
            return None

    def cancel(self, task_id: str) -> bool:
        """Cancel a queued task."""
        with self._lock:
            if task_id in self._task_map:
                task = self._task_map[task_id]
                task.state = TaskState.CANCELLED
                del self._task_map[task_id]
                return True
            return False

    def size(self) -> int:
        with self._lock:
            return len(self._task_map)

    def is_empty(self) -> bool:
        return self.size() == 0

    def clear(self) -> int:
        """Clear all tasks. Returns number cleared."""
        with self._lock:
            count = len(self._task_map)
            self._heap.clear()
            self._task_map.clear()
            return count


# =============================================================================
# WORKER & WORKER POOL
# =============================================================================

class Worker:
    """A worker thread for executing tasks."""

    def __init__(self, worker_id: str, task_queue: TaskQueue):
        self.id = worker_id
        self.state = WorkerState.IDLE
        self._queue = task_queue
        self._current_task: Optional[Task] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._results: List[TaskResult] = []

    def start(self):
        """Start the worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.state = WorkerState.TERMINATED

    def _run(self):
        """Main worker loop."""
        while not self._stop_event.is_set():
            task = self._queue.pop()
            if task is None:
                time.sleep(0.01)  # Prevent busy-wait
                continue

            self.state = WorkerState.BUSY
            self._current_task = task
            result = self._execute_task(task)
            self._results.append(result)
            self._current_task = None
            self.state = WorkerState.IDLE

    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        task.state = TaskState.RUNNING
        task.started_at = time.time()
        start = time.time()

        try:
            if task.func:
                output = task.func(*task.args, **task.kwargs)
            else:
                output = None

            task.state = TaskState.COMPLETED
            task.completed_at = time.time()
            task.result = output

            return TaskResult(
                task_id=task.id,
                success=True,
                output=output,
                execution_time_ms=int((time.time() - start) * 1000),
            )

        except Exception as e:
            task.retries += 1
            if task.retries < task.max_retries:
                task.state = TaskState.PENDING
                self._queue.push(task)  # Retry
            else:
                task.state = TaskState.FAILED
                task.error = str(e)

            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time_ms=int((time.time() - start) * 1000),
                retries_used=task.retries,
            )


class WorkerPool:
    """
    Pool of worker threads with PHI-optimal sizing.

    Default size is based on golden ratio:
    - Base workers: PHI^2 ≈ 2.618 → 3
    - Max workers: PHI^4 ≈ 6.854 → 7
    """

    def __init__(self, min_workers: int = 3, max_workers: int = 7):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self._queue = TaskQueue()
        self._workers: List[Worker] = []
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Start the worker pool."""
        with self._lock:
            if self._running:
                return
            self._running = True
            for i in range(self.min_workers):
                worker = Worker(f"worker-{i}", self._queue)
                worker.start()
                self._workers.append(worker)

    def stop(self):
        """Stop all workers."""
        with self._lock:
            self._running = False
            for worker in self._workers:
                worker.stop()
            self._workers.clear()

    def submit(self, task: Task) -> bool:
        """Submit a task to the pool."""
        if not self._running:
            return False
        return self._queue.push(task)

    def scale_up(self) -> bool:
        """Add a worker if below max."""
        with self._lock:
            if len(self._workers) >= self.max_workers:
                return False
            worker = Worker(f"worker-{len(self._workers)}", self._queue)
            worker.start()
            self._workers.append(worker)
            return True

    def scale_down(self) -> bool:
        """Remove a worker if above min."""
        with self._lock:
            if len(self._workers) <= self.min_workers:
                return False
            worker = self._workers.pop()
            worker.stop()
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            idle = sum(1 for w in self._workers if w.state == WorkerState.IDLE)
            busy = sum(1 for w in self._workers if w.state == WorkerState.BUSY)
            return {
                "total_workers": len(self._workers),
                "idle_workers": idle,
                "busy_workers": busy,
                "queue_size": self._queue.size(),
                "running": self._running,
            }


# =============================================================================
# ADAPTIVE SCHEDULER
# =============================================================================

class AdaptiveScheduler:
    """
    Resource-aware task scheduler with PHI-based allocation.

    Features:
    - Priority-based scheduling
    - Resource tracking per task
    - Hardware-aware task selection
    - PHI-optimal batch sizing
    """

    def __init__(self):
        self._queue = TaskQueue()
        self._profiles: Dict[str, List[ResourceProfile]] = {}
        self._hardware: Optional[HardwareSnapshot] = None
        self._completed: Set[str] = set()
        self._cycle = 0
        self._lock = threading.Lock()

    def submit(self, task: Task) -> bool:
        """Submit a task for scheduling."""
        return self._queue.push(task)

    def get_next_task(self, hardware: HardwareSnapshot = None) -> Optional[Task]:
        """
        Get next task considering hardware constraints.

        Uses PHI-based selection when multiple tasks available.
        """
        if hardware:
            self._hardware = hardware

        task = self._queue.peek()
        if task is None:
            return None

        # Check if task fits hardware constraints
        if self._hardware and not self._can_fit_task(task):
            # Try to find a smaller task
            # For now, just return None if top task doesn't fit
            return None

        return self._queue.pop()

    def _can_fit_task(self, task: Task) -> bool:
        """Check if task fits current hardware state."""
        if self._hardware is None:
            return True

        # Estimate resource needs from profile
        profile = self._get_average_profile(task.name)
        if profile is None:
            return True  # No history, assume it fits

        # Check VRAM with safety margin (BETA)
        needed_mb = profile.vram_used_mb * (1 + BETA)
        return self._hardware.vram_free_mb >= needed_mb

    def record_completion(self, task: Task, profile: ResourceProfile):
        """Record task completion for learning."""
        with self._lock:
            if task.name not in self._profiles:
                self._profiles[task.name] = []
            self._profiles[task.name].append(profile)
            # Keep only last 10 profiles per task
            self._profiles[task.name] = self._profiles[task.name][-10:]
            self._completed.add(task.id)

    def _get_average_profile(self, task_name: str) -> Optional[ResourceProfile]:
        """Get average resource profile for a task type."""
        profiles = self._profiles.get(task_name, [])
        if not profiles:
            return None

        return ResourceProfile(
            name=task_name,
            vram_used_mb=sum(p.vram_used_mb for p in profiles) / len(profiles),
            time_seconds=sum(p.time_seconds for p in profiles) / len(profiles),
            tokens_generated=int(sum(p.tokens_generated for p in profiles) / len(profiles)),
            tokens_per_second=sum(p.tokens_per_second for p in profiles) / len(profiles),
        )

    def start_cycle(self):
        """Start a new scheduling cycle."""
        self._cycle += 1
        self._completed.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "cycle": self._cycle,
            "queue_size": self._queue.size(),
            "completed_this_cycle": len(self._completed),
            "known_task_types": len(self._profiles),
        }


# =============================================================================
# PROCESS MANAGER
# =============================================================================

class ProcessManager:
    """
    High-level process/task lifecycle manager.

    Combines scheduler and worker pool for complete task management.
    """

    def __init__(self, pool_size: int = 4):
        self.scheduler = AdaptiveScheduler()
        self.pool = WorkerPool(min_workers=pool_size, max_workers=pool_size * 2)
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start the process manager."""
        self.pool.start()

    def stop(self):
        """Stop the process manager."""
        self.pool.stop()

    def create_task(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> Task:
        """Create and register a new task."""
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
        )
        with self._lock:
            self._tasks[task.id] = task
        return task

    def submit(self, task: Task) -> bool:
        """Submit a task for execution."""
        success = self.pool.submit(task)
        if success:
            self.scheduler.submit(task)
        return success

    def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.state = TaskState.CANCELLED
                return True
            return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result."""
        return self._results.get(task_id)

    def list_tasks(self, state: TaskState = None) -> List[Task]:
        """List tasks, optionally filtered by state."""
        with self._lock:
            tasks = list(self._tasks.values())
            if state:
                tasks = [t for t in tasks if t.state == state]
            return tasks

    def get_stats(self) -> Dict[str, Any]:
        """Get process manager statistics."""
        return {
            "total_tasks": len(self._tasks),
            "scheduler": self.scheduler.get_stats(),
            "pool": self.pool.get_stats(),
        }


# =============================================================================
# MODE REGISTRY & BASE MODE
# =============================================================================

class BaseMode(ABC):
    """Abstract base class for operational modes."""

    def __init__(self, name: str, mode_type: ModeType):
        self.name = name
        self.mode_type = mode_type
        self._active = False

    @abstractmethod
    def enter(self, context: Dict[str, Any]) -> bool:
        """Enter this mode. Returns success."""
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Optional[ModeTransition]:
        """Execute one cycle. Returns transition request or None."""
        pass

    @abstractmethod
    def exit(self, context: Dict[str, Any]) -> bool:
        """Exit this mode. Returns success."""
        pass


class ModeRegistry:
    """Registry for operational modes."""

    def __init__(self):
        self._modes: Dict[str, Type[BaseMode]] = {}
        self._instances: Dict[str, BaseMode] = {}

    def register(self, name: str, mode_class: Type[BaseMode]):
        """Register a mode class."""
        self._modes[name] = mode_class

    def get(self, name: str) -> Optional[BaseMode]:
        """Get or create a mode instance."""
        if name not in self._modes:
            return None
        if name not in self._instances:
            self._instances[name] = self._modes[name](name, ModeType.SOVEREIGN)
        return self._instances[name]

    def list_modes(self) -> List[str]:
        """List registered mode names."""
        return list(self._modes.keys())


# =============================================================================
# CORTEX SWITCHBOARD
# =============================================================================

class CortexSwitchboard:
    """
    Hot-swap between operational modes without kernel restart.

    Maintains persistent context across mode transitions.
    """

    def __init__(self, registry: ModeRegistry, start_mode: str = "sovereign"):
        self.registry = registry
        self.current_mode_name = start_mode
        self._context: Dict[str, Any] = {}
        self._running = False
        self._transition_history: List[ModeTransition] = []

    def run(self, initial_context: Dict[str, Any] = None):
        """Main orchestration loop."""
        self._context = initial_context or {}
        self._running = True

        while self._running:
            mode = self.registry.get(self.current_mode_name)
            if mode is None:
                logger.error(f"Unknown mode: {self.current_mode_name}")
                break

            # Enter mode
            if not mode.enter(self._context):
                logger.error(f"Failed to enter mode: {self.current_mode_name}")
                break

            # Execute mode cycles until transition
            transition = None
            while self._running and transition is None:
                try:
                    transition = mode.execute(self._context)
                except Exception as e:
                    logger.error(f"Mode execution error: {e}")
                    break

            # Exit mode
            mode.exit(self._context)

            # Handle transition
            if transition:
                self._transition_history.append(transition)
                self._context.update(transition.payload)
                self.current_mode_name = transition.to_mode
            else:
                break

    def stop(self):
        """Stop the switchboard."""
        self._running = False

    def request_transition(self, to_mode: str, reason: str = "", payload: Dict = None):
        """Request a mode transition."""
        transition = ModeTransition(
            from_mode=self.current_mode_name,
            to_mode=to_mode,
            reason=reason,
            payload=payload or {},
        )
        self._transition_history.append(transition)
        self.current_mode_name = to_mode


# =============================================================================
# KERNEL SUBSTRATE
# =============================================================================

class KernelSubstrate:
    """
    Central orchestrator connecting all major systems.

    Level 6 ASI Architecture with 22 tiers:
    - Tier 1: Budget & Safety
    - Tier 2-4: Routing & Vision
    - Tier 5-8: Temporal & Evaluation
    - Tier 9-12: Agents & Archival
    - Tier 13-18: Level 6 ASI Components
    - Tier 19-22: Advanced Systems
    """

    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self._components: Dict[str, Any] = {}
        self._status: Dict[str, bool] = {}

        # Initialize all tiers
        self._init_tier_1_budget()
        self._init_tier_2_safety()
        self._init_tier_3_routing()
        self._init_tier_4_temporal()
        self._init_tier_5_evaluation()
        self._init_tier_6_agents()

        logger.info(f"KernelSubstrate initialized with {len(self._components)} components")

    def _init_tier_1_budget(self):
        """Tier 1: Budget & Resource System."""
        self._components["budget_ledger"] = {}
        self._status["tier_1"] = True

    def _init_tier_2_safety(self):
        """Tier 2: Hardware Safety."""
        self._components["safety_governor"] = {"enabled": True}
        self._components["npu_available"] = False
        self._status["tier_2"] = True

    def _init_tier_3_routing(self):
        """Tier 3: Model Routing."""
        self._components["router"] = None
        self._components["neuronic_router"] = None
        self._status["tier_3"] = True

    def _init_tier_4_temporal(self):
        """Tier 4: Temporal & Pulse System."""
        self._components["pulse"] = {"beat": 0, "cycle": 0}
        self._status["tier_4"] = True

    def _init_tier_5_evaluation(self):
        """Tier 5: Evaluation & Compliance."""
        self._components["evaluator"] = None
        self._components["compliance"] = None
        self._status["tier_5"] = True

    def _init_tier_6_agents(self):
        """Tier 6: Agent Systems."""
        self._components["professor"] = None
        self._components["alpha"] = None
        self._status["tier_6"] = True

    def get_component(self, name: str) -> Any:
        """Get a kernel component by name."""
        return self._components.get(name)

    def set_component(self, name: str, component: Any):
        """Set a kernel component."""
        self._components[name] = component

    def get_status(self) -> Dict[str, Any]:
        """Get status of all tiers."""
        return {
            "tiers": self._status,
            "components": len(self._components),
            "repo_root": str(self.repo_root),
        }

    def shutdown(self):
        """Clean shutdown of all services."""
        logger.info("Initiating kernel substrate shutdown...")
        self._components.clear()
        logger.info("Kernel substrate shutdown complete")


# =============================================================================
# KERNEL (UNIFIED INTERFACE)
# =============================================================================

class Kernel:
    """
    Unified kernel interface for SPIO.

    Provides access to:
    - Process management
    - Task scheduling
    - Mode switching
    - Resource monitoring
    """

    def __init__(self, repo_root: str = "."):
        self.substrate = KernelSubstrate(repo_root)
        self.process_manager = ProcessManager()
        self.scheduler = AdaptiveScheduler()
        self.mode_registry = ModeRegistry()
        self.switchboard = CortexSwitchboard(self.mode_registry)
        self._running = False

    def boot(self) -> bool:
        """Boot the kernel."""
        try:
            self.process_manager.start()
            self._running = True
            logger.info("Kernel booted successfully")
            return True
        except Exception as e:
            logger.error(f"Kernel boot failed: {e}")
            return False

    def shutdown(self) -> bool:
        """Shutdown the kernel."""
        try:
            self._running = False
            self.switchboard.stop()
            self.process_manager.stop()
            self.substrate.shutdown()
            logger.info("Kernel shutdown complete")
            return True
        except Exception as e:
            logger.error(f"Kernel shutdown error: {e}")
            return False

    def submit_task(
        self,
        name: str,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> Optional[Task]:
        """Submit a task for execution."""
        if not self._running:
            return None
        task = self.process_manager.create_task(name, func, args, kwargs, priority)
        self.process_manager.submit(task)
        return task

    def get_stats(self) -> Dict[str, Any]:
        """Get kernel statistics."""
        return {
            "running": self._running,
            "substrate": self.substrate.get_status(),
            "process_manager": self.process_manager.get_stats(),
            "scheduler": self.scheduler.get_stats(),
            "modes": self.mode_registry.list_modes(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_kernel: Optional[Kernel] = None


def create_kernel(repo_root: str = ".") -> Kernel:
    """Create a new kernel instance."""
    return Kernel(repo_root)


def create_scheduler() -> AdaptiveScheduler:
    """Create a new adaptive scheduler."""
    return AdaptiveScheduler()


def create_process_manager(pool_size: int = 4) -> ProcessManager:
    """Create a new process manager."""
    return ProcessManager(pool_size)


def get_kernel() -> Kernel:
    """Get or create the global kernel."""
    global _kernel
    if _kernel is None:
        _kernel = Kernel()
    return _kernel
