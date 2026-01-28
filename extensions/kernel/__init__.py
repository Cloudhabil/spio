"""
Kernel Extension - Process Scheduler and Substrate

Level 6 ASI Architecture - Core kernel components:

Components:
1. KernelSubstrate (22 tiers) - Central orchestrator connecting all systems
2. CortexSwitchboard - Hot-swap mode transitions
3. AdaptiveScheduler - Resource-aware task scheduling
4. ProcessManager - Task lifecycle and priority queues
5. TaskQueue - Priority-based task execution
6. WorkerPool - Thread/process pool management

Reference: CLI-main/src/core/kernel/*
"""

from .kernel_core import (
    # ========================================================================
    # ENUMS
    # ========================================================================
    TaskPriority,
    TaskState,
    WorkerState,
    ModeType,

    # ========================================================================
    # DATA CLASSES
    # ========================================================================
    Task,
    TaskResult,
    HardwareSnapshot,
    ResourceProfile,
    ModeTransition,

    # ========================================================================
    # KERNEL SUBSTRATE
    # ========================================================================
    KernelSubstrate,

    # ========================================================================
    # SWITCHBOARD
    # ========================================================================
    CortexSwitchboard,

    # ========================================================================
    # SCHEDULER
    # ========================================================================
    AdaptiveScheduler,
    TaskQueue,

    # ========================================================================
    # PROCESS MANAGER
    # ========================================================================
    ProcessManager,
    Worker,
    WorkerPool,

    # ========================================================================
    # MODE REGISTRY
    # ========================================================================
    ModeRegistry,
    BaseMode,

    # ========================================================================
    # KERNEL INTERFACE
    # ========================================================================
    Kernel,

    # ========================================================================
    # FACTORY FUNCTIONS
    # ========================================================================
    create_kernel,
    create_scheduler,
    create_process_manager,
    get_kernel,
)

__all__ = [
    # Enums
    "TaskPriority",
    "TaskState",
    "WorkerState",
    "ModeType",

    # Data classes
    "Task",
    "TaskResult",
    "HardwareSnapshot",
    "ResourceProfile",
    "ModeTransition",

    # Substrate
    "KernelSubstrate",

    # Switchboard
    "CortexSwitchboard",

    # Scheduler
    "AdaptiveScheduler",
    "TaskQueue",

    # Process Manager
    "ProcessManager",
    "Worker",
    "WorkerPool",

    # Mode Registry
    "ModeRegistry",
    "BaseMode",

    # Main interface
    "Kernel",

    # Factories
    "create_kernel",
    "create_scheduler",
    "create_process_manager",
    "get_kernel",
]
