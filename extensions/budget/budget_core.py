"""
Budget Extension - Resource Allocation and Management

Ported from:
- CLI-main/src/core/budget_ledger.py
- CLI-main/src/core/dynamic_budget_orchestrator.py
- CLI-main/src/core/kernel/budget_service.py
- CLI-main/src/core/resource_analyzer.py

Implements:
- BudgetLedger: Thread-safe global token allocation tracker
- BudgetOrchestrator: Dynamic budget computation
- BudgetService: Kernel-level resource allocation
- ResourceAnalyzer: Tiered data access and context windowing
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# BUDGET ALLOCATION
# =============================================================================

@dataclass
class BudgetAllocation:
    """
    Represents a single token allocation.

    Attributes:
        task_id: Unique identifier for the task
        agent: Agent name (professor, alpha, gpia, etc.)
        model: Model ID (e.g., codegemma:latest)
        tokens: Number of tokens allocated
        timestamp: When the allocation was made
        status: Current status (pending, active, completed, failed)
    """
    task_id: str
    agent: str
    model: str
    tokens: int
    timestamp: float
    status: str  # "pending", "active", "completed", "failed"

    def is_active(self) -> bool:
        """Check if allocation is currently active."""
        return self.status in ["pending", "active"]


# =============================================================================
# BUDGET LEDGER
# =============================================================================

class BudgetLedger:
    """
    Thread-safe global budget tracker.

    Prevents concurrent overallocation by tracking all active allocations
    and enforcing global limits. Uses transaction-style budget enforcement.
    """

    def __init__(self, cleanup_threshold_seconds: int = 300):
        """
        Initialize the budget ledger.

        Args:
            cleanup_threshold_seconds: Remove allocations older than this
        """
        self._allocations: Dict[str, BudgetAllocation] = {}
        self._lock = threading.RLock()
        self._cleanup_threshold = cleanup_threshold_seconds

    def reserve(self, task_id: str, agent: str, model: str, tokens: int) -> bool:
        """
        Reserve tokens for a task.

        Args:
            task_id: Unique task identifier
            agent: Agent name
            model: Model ID
            tokens: Number of tokens to reserve

        Returns:
            True if reservation successful
        """
        with self._lock:
            self._allocations[task_id] = BudgetAllocation(
                task_id=task_id,
                agent=agent,
                model=model,
                tokens=tokens,
                timestamp=time.time(),
                status="pending"
            )
            return True

    def activate(self, task_id: str) -> None:
        """Mark allocation as active (execution started)."""
        with self._lock:
            if task_id in self._allocations:
                self._allocations[task_id].status = "active"

    def release(self, task_id: str, status: str = "completed") -> None:
        """
        Release allocated tokens.

        Args:
            task_id: Task identifier
            status: Final status (completed, failed)
        """
        with self._lock:
            if task_id in self._allocations:
                self._allocations[task_id].status = status
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove old allocations to prevent memory leak."""
        now = time.time()
        self._allocations = {
            k: v for k, v in self._allocations.items()
            if (now - v.timestamp) < self._cleanup_threshold or v.is_active()
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        with self._lock:
            by_agent: Dict[str, int] = {}
            by_model: Dict[str, int] = {}
            total_active = 0
            by_status: Dict[str, int] = {}

            for alloc in self._allocations.values():
                if alloc.is_active():
                    total_active += alloc.tokens
                    by_agent[alloc.agent] = by_agent.get(alloc.agent, 0) + alloc.tokens
                    by_model[alloc.model] = by_model.get(alloc.model, 0) + alloc.tokens

                by_status[alloc.status] = by_status.get(alloc.status, 0) + 1

            return {
                "total_active_tokens": total_active,
                "by_agent": by_agent,
                "by_model": by_model,
                "by_status": by_status,
                "total_allocations": len(self._allocations)
            }

    def get_agent_tokens(self, agent: str) -> int:
        """Get total active tokens for an agent."""
        with self._lock:
            return sum(
                a.tokens for a in self._allocations.values()
                if a.agent == agent and a.is_active()
            )

    def get_model_tokens(self, model: str) -> int:
        """Get total active tokens for a model."""
        with self._lock:
            return sum(
                a.tokens for a in self._allocations.values()
                if a.model == model and a.is_active()
            )

    def get_allocation(self, task_id: str) -> Optional[BudgetAllocation]:
        """Get allocation details by task ID."""
        with self._lock:
            return self._allocations.get(task_id)


# =============================================================================
# BUDGET SETTINGS
# =============================================================================

def _env_bool(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class BudgetSettings:
    """
    Budget configuration from environment variables.

    Profiles:
        - safe: 0.6x tokens (conservative)
        - fast: 0.8x tokens (speed-focused)
        - balanced: 1.0x tokens (default)
        - quality: 1.25x tokens (quality-focused)
    """
    enabled: bool = True
    profile: str = "balanced"
    min_tokens: int = 128
    max_tokens: int = 4096
    allow_upscale: bool = False
    prompt_ratio: float = 1.2
    tokens_per_gb_ram: int = 256
    tokens_per_gb_vram: int = 384
    reserve_ram_mb: int = 2048
    reserve_vram_mb: int = 1024
    resource_ttl: int = 20
    log_decisions: bool = False
    profile_factors: Dict[str, float] = field(default_factory=lambda: {
        "safe": 0.6,
        "fast": 0.8,
        "balanced": 1.0,
        "quality": 1.25,
    })

    @classmethod
    def from_env(cls) -> BudgetSettings:
        """Create settings from environment variables."""
        return cls(
            enabled=_env_bool("GPIA_DYNAMIC_BUDGET", "1"),
            profile=os.getenv("GPIA_BUDGET_PROFILE", "balanced"),
            min_tokens=_env_int("GPIA_BUDGET_MIN_TOKENS", 128),
            max_tokens=_env_int("GPIA_BUDGET_MAX_TOKENS", 4096),
            allow_upscale=_env_bool("GPIA_BUDGET_ALLOW_UPSCALE", "0"),
            prompt_ratio=_env_float("GPIA_BUDGET_PROMPT_RATIO", 1.2),
            tokens_per_gb_ram=_env_int("GPIA_BUDGET_TOKENS_PER_GB_RAM", 256),
            tokens_per_gb_vram=_env_int("GPIA_BUDGET_TOKENS_PER_GB_VRAM", 384),
            reserve_ram_mb=_env_int("GPIA_BUDGET_RESERVE_RAM_MB", 2048),
            reserve_vram_mb=_env_int("GPIA_BUDGET_RESERVE_VRAM_MB", 1024),
            resource_ttl=_env_int("GPIA_BUDGET_RESOURCE_TTL", 20),
            log_decisions=_env_bool("GPIA_BUDGET_LOG", "0"),
        )


# =============================================================================
# BUDGET ORCHESTRATOR
# =============================================================================

class BudgetOrchestrator:
    """
    Dynamic budget computation based on system resources.

    Features:
        - Profile-based scaling (safe, fast, balanced, quality)
        - Resource-aware limits (RAM, VRAM)
        - Model-specific factors
        - Prompt-based caps
    """

    def __init__(self, settings: Optional[BudgetSettings] = None):
        self.settings = settings or BudgetSettings.from_env()
        self._resource_cache: Dict[str, Any] = {"timestamp": 0.0, "data": None}

    def _get_memory_stats_mb(self) -> Dict[str, Optional[int]]:
        """Get RAM statistics."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {
                "total_mb": int(vm.total / (1024 * 1024)),
                "free_mb": int(vm.available / (1024 * 1024)),
            }
        except ImportError:
            pass

        # Fallback for Windows without psutil
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_OperatingSystem | "
                 "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                total_kb = int(data.get("TotalVisibleMemorySize", 0))
                free_kb = int(data.get("FreePhysicalMemory", 0))
                return {"total_mb": total_kb // 1024, "free_mb": free_kb // 1024}
        except Exception:
            pass

        return {"total_mb": None, "free_mb": None}

    def _get_vram_stats_mb(self) -> Dict[str, Optional[int]]:
        """Get VRAM statistics via nvidia-smi."""
        if shutil.which("nvidia-smi") is None:
            return {"total_mb": None, "free_mb": None}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                line = (result.stdout or "").strip().splitlines()[0]
                total_mb, used_mb = [int(x.strip()) for x in line.split(",")[:2]]
                return {"total_mb": total_mb, "free_mb": max(total_mb - used_mb, 0)}
        except Exception:
            pass
        return {"total_mb": None, "free_mb": None}

    def _get_resource_snapshot(self) -> Dict[str, Optional[int]]:
        """Get cached resource snapshot."""
        now = time.time()
        cached = self._resource_cache.get("data")
        if cached and (now - self._resource_cache.get("timestamp", 0) < self.settings.resource_ttl):
            return cached

        mem = self._get_memory_stats_mb()
        vram = self._get_vram_stats_mb()
        snapshot = {
            "ram_total_mb": mem.get("total_mb"),
            "ram_free_mb": mem.get("free_mb"),
            "vram_total_mb": vram.get("total_mb"),
            "vram_free_mb": vram.get("free_mb"),
            "timestamp": int(now),
        }
        self._resource_cache["timestamp"] = now
        self._resource_cache["data"] = snapshot
        return snapshot

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        """Estimate token count from prompt length."""
        if not prompt:
            return 0
        return max(1, int(len(prompt) / 4))

    def _model_factor(self, model_id: Optional[str]) -> float:
        """Get model-specific scaling factor."""
        if not model_id:
            return 1.0
        model_id = model_id.lower()
        if "gpt-oss" in model_id or "20b" in model_id:
            return 0.6
        if "deepseek" in model_id:
            return 0.8
        if "qwen" in model_id:
            return 0.9
        if "llava" in model_id:
            return 0.8
        return 1.0

    def compute(
        self,
        prompt: str,
        requested_tokens: int,
        model_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Compute effective token budget.

        Args:
            prompt: The prompt text
            requested_tokens: Requested token count
            model_id: Model identifier
            profile: Budget profile (safe, fast, balanced, quality)

        Returns:
            Tuple of (effective_tokens, details)
        """
        settings = self.settings
        requested_tokens = max(1, int(requested_tokens or settings.min_tokens))

        if not settings.enabled:
            return requested_tokens, {
                "reason": "disabled",
                "requested_tokens": requested_tokens,
            }

        profile = profile or settings.profile
        profile_factor = settings.profile_factors.get(profile, 1.0)
        prompt_tokens = self._estimate_prompt_tokens(prompt)
        prompt_cap = max(settings.min_tokens, int(prompt_tokens * settings.prompt_ratio))

        snapshot = self._get_resource_snapshot()
        model_factor = self._model_factor(model_id)
        tokens_per_gb_ram = int(settings.tokens_per_gb_ram * model_factor)
        tokens_per_gb_vram = int(settings.tokens_per_gb_vram * model_factor)

        resource_caps = []

        ram_mb = snapshot.get("ram_free_mb") or snapshot.get("ram_total_mb")
        if ram_mb:
            usable_ram_mb = max(int(ram_mb) - settings.reserve_ram_mb, 0)
            resource_caps.append(int((usable_ram_mb / 1024.0) * tokens_per_gb_ram))

        vram_mb = snapshot.get("vram_free_mb") or snapshot.get("vram_total_mb")
        if vram_mb:
            usable_vram_mb = max(int(vram_mb) - settings.reserve_vram_mb, 0)
            resource_caps.append(int((usable_vram_mb / 1024.0) * tokens_per_gb_vram))

        resource_cap = min(resource_caps) if resource_caps else None
        adjusted = int(requested_tokens * profile_factor)

        caps = [adjusted, prompt_cap]
        if resource_cap is not None:
            caps.append(resource_cap)
        if settings.max_tokens > 0:
            caps.append(settings.max_tokens)
        if not settings.allow_upscale:
            caps.append(requested_tokens)

        effective = max(settings.min_tokens, min(caps))

        details = {
            "profile": profile,
            "profile_factor": profile_factor,
            "requested_tokens": requested_tokens,
            "adjusted_tokens": adjusted,
            "prompt_tokens": prompt_tokens,
            "prompt_cap": prompt_cap,
            "resource_cap": resource_cap,
            "min_tokens": settings.min_tokens,
            "max_tokens": settings.max_tokens,
            "allow_upscale": settings.allow_upscale,
            "model_factor": model_factor,
            "resource_snapshot": snapshot,
        }

        if settings.log_decisions:
            logger.info(
                f"[budget] requested={requested_tokens} effective={effective} "
                f"profile={profile} resource_cap={resource_cap}"
            )

        return effective, details


# =============================================================================
# RESOURCE SNAPSHOT
# =============================================================================

@dataclass
class ResourceSnapshot:
    """
    Current system resource state.

    Attributes:
        timestamp: When snapshot was taken
        cpu_percent: CPU utilization percentage
        ram_total_mb: Total RAM in MB
        ram_free_mb: Free RAM in MB
        ram_used_mb: Used RAM in MB
        vram_total_mb: Total VRAM in MB
        vram_free_mb: Free VRAM in MB
        vram_used_mb: Used VRAM in MB
        disk_read_mbps: Disk read rate in MB/s
        disk_write_mbps: Disk write rate in MB/s
        npu_available: Whether NPU is available
    """
    timestamp: float
    cpu_percent: float = 0.0
    ram_total_mb: int = 0
    ram_free_mb: int = 0
    ram_used_mb: int = 0
    vram_total_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    vram_used_mb: Optional[int] = None
    disk_read_mbps: float = 0.0
    disk_write_mbps: float = 0.0
    npu_available: bool = False

    @property
    def ram_util(self) -> float:
        """RAM utilization ratio (0.0-1.0)."""
        return self.ram_used_mb / max(self.ram_total_mb, 1)

    @property
    def vram_util(self) -> Optional[float]:
        """VRAM utilization ratio (0.0-1.0)."""
        if self.vram_total_mb and self.vram_used_mb is not None:
            return self.vram_used_mb / max(self.vram_total_mb, 1)
        return None


# =============================================================================
# SAFETY LIMITS
# =============================================================================

@dataclass
class SafetyLimits:
    """
    Hard safety limits to prevent hardware damage.

    SUBSTRATE EQUILIBRIUM (v0.5.0):
    VRAM limits are calibrated to the 9750MB cliff to prevent
    "driver juggling" instability when Windows DWM contends for memory.

    Physical substrate (RTX 4070 SUPER 12GB):
        - 12288 MB total VRAM
        - ~1700 MB reserved by Windows DWM
        - ~850 MB safety buffer
        - 9750 MB = operational ceiling (79.3% of 12288)
    """
    # HARDWARE SOVEREIGNTY - calibrated to physical substrate
    max_vram_util: float = 0.793         # 9750/12288 = 79.3%
    max_vram_mb: int = 9750              # Absolute cliff
    vram_reserve_mb: int = 2538          # DWM + buffer

    # Other limits
    max_ram_util: float = 0.90           # 90%
    max_cpu_util: float = 0.95           # 95%
    max_disk_write_mbps: float = 500.0   # Prevent SSD wear
    ram_reserve_mb: int = 2048           # Always keep 2GB RAM free


# =============================================================================
# BUDGET SERVICE
# =============================================================================

class BudgetService:
    """
    Kernel service for resource allocation and safety enforcement.

    Features:
        - Real-time resource monitoring (CPU, GPU, NPU, RAM, VRAM, Disk I/O)
        - Hard safety limits to prevent GPU damage
        - Transaction-style allocation tracking
        - Integration with budget orchestrator
    """

    def __init__(self):
        """Initialize budget service."""
        self.ledger = BudgetLedger()
        self.orchestrator = BudgetOrchestrator()
        self.limits = SafetyLimits()
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._last_snapshot_time = 0.0
        self._snapshot_ttl = 2.0

        # Disk I/O tracking
        self._last_disk_io: Optional[Tuple[int, int]] = None
        self._last_disk_io_time = 0.0

        # Emergency shutdown flag
        self._emergency_shutdown = False

    def get_resource_snapshot(self, force_refresh: bool = False) -> ResourceSnapshot:
        """
        Get current resource state with caching.

        Args:
            force_refresh: Skip cache and get fresh data

        Returns:
            ResourceSnapshot with current resource state
        """
        now = time.time()
        if not force_refresh and self._last_snapshot and \
                (now - self._last_snapshot_time) < self._snapshot_ttl:
            return self._last_snapshot

        # Get RAM/CPU
        cpu_percent = 0.0
        ram_total_mb = 0
        ram_used_mb = 0
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            ram_total_mb = int(vm.total / (1024 * 1024))
            ram_used_mb = int(vm.used / (1024 * 1024))
        except ImportError:
            pass

        # Get VRAM
        vram_total_mb = None
        vram_used_mb = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                line = (result.stdout or "").strip().split("\n")[0]
                parts = [int(x.strip()) for x in line.split(",")[:2]]
                if len(parts) == 2:
                    vram_total_mb, vram_used_mb = parts
        except Exception:
            pass

        # Get disk I/O
        disk_read_mbps, disk_write_mbps = self._get_disk_io_rates()

        snapshot = ResourceSnapshot(
            timestamp=now,
            cpu_percent=cpu_percent,
            ram_total_mb=ram_total_mb,
            ram_free_mb=ram_total_mb - ram_used_mb,
            ram_used_mb=ram_used_mb,
            vram_total_mb=vram_total_mb,
            vram_free_mb=(vram_total_mb - vram_used_mb) if vram_total_mb and vram_used_mb else None,
            vram_used_mb=vram_used_mb,
            disk_read_mbps=disk_read_mbps,
            disk_write_mbps=disk_write_mbps,
            npu_available=False,
        )

        self._last_snapshot = snapshot
        self._last_snapshot_time = now
        return snapshot

    def _get_disk_io_rates(self) -> Tuple[float, float]:
        """Calculate disk I/O rates in MB/s."""
        try:
            import psutil
            counters = psutil.disk_io_counters()
            now = time.time()

            if self._last_disk_io is None:
                self._last_disk_io = (counters.read_bytes, counters.write_bytes)
                self._last_disk_io_time = now
                return 0.0, 0.0

            elapsed = now - self._last_disk_io_time
            if elapsed < 0.1:
                return 0.0, 0.0

            read_delta = counters.read_bytes - self._last_disk_io[0]
            write_delta = counters.write_bytes - self._last_disk_io[1]

            read_mbps = (read_delta / elapsed) / (1024 * 1024)
            write_mbps = (write_delta / elapsed) / (1024 * 1024)

            self._last_disk_io = (counters.read_bytes, counters.write_bytes)
            self._last_disk_io_time = now

            return max(0.0, read_mbps), max(0.0, write_mbps)
        except Exception:
            return 0.0, 0.0

    def check_safety(self, snapshot: Optional[ResourceSnapshot] = None) -> Tuple[bool, str]:
        """
        Check if current resource state is safe.

        Args:
            snapshot: ResourceSnapshot to check (or None to get fresh)

        Returns:
            Tuple of (is_safe, reason)
        """
        if snapshot is None:
            snapshot = self.get_resource_snapshot(force_refresh=True)

        # Check VRAM (CRITICAL - prevents GPU damage and DWM contention)
        if snapshot.vram_util is not None and snapshot.vram_util >= self.limits.max_vram_util:
            self._emergency_shutdown = True
            return False, (
                f"VRAM critical: {snapshot.vram_util*100:.1f}% >= "
                f"{self.limits.max_vram_util*100:.1f}% (cliff)"
            )

        # Absolute MB check
        if snapshot.vram_used_mb is not None and \
                snapshot.vram_used_mb >= self.limits.max_vram_mb:
            self._emergency_shutdown = True
            return False, (
                f"VRAM CLIFF BREACH: {snapshot.vram_used_mb}MB >= "
                f"{self.limits.max_vram_mb}MB (DWM contention zone)"
            )

        if snapshot.vram_free_mb is not None and \
                snapshot.vram_free_mb < self.limits.vram_reserve_mb:
            return False, (
                f"VRAM reserve breach: {snapshot.vram_free_mb}MB < "
                f"{self.limits.vram_reserve_mb}MB"
            )

        # Check RAM
        if snapshot.ram_util >= self.limits.max_ram_util:
            return False, (
                f"RAM critical: {snapshot.ram_util*100:.1f}% >= "
                f"{self.limits.max_ram_util*100:.0f}%"
            )

        # Check CPU
        if snapshot.cpu_percent >= self.limits.max_cpu_util * 100:
            return False, (
                f"CPU critical: {snapshot.cpu_percent:.1f}% >= "
                f"{self.limits.max_cpu_util*100:.0f}%"
            )

        # Check disk write rate
        if snapshot.disk_write_mbps > self.limits.max_disk_write_mbps:
            return False, (
                f"Disk write critical: {snapshot.disk_write_mbps:.1f} MB/s > "
                f"{self.limits.max_disk_write_mbps:.0f} MB/s"
            )

        return True, "ok"

    def request_allocation(
        self,
        task_id: str,
        agent: str,
        model: str,
        prompt: str,
        requested_tokens: int
    ) -> Tuple[bool, int, str]:
        """
        Request token allocation for a task.

        Args:
            task_id: Unique task identifier
            agent: Agent name
            model: Model ID
            prompt: The prompt text
            requested_tokens: Requested token count

        Returns:
            Tuple of (approved, allocated_tokens, reason)
        """
        # Check safety first
        snapshot = self.get_resource_snapshot(force_refresh=True)
        is_safe, safety_reason = self.check_safety(snapshot)

        if not is_safe:
            return False, 0, f"Safety check failed: {safety_reason}"

        # Compute budget
        effective_tokens, _ = self.orchestrator.compute(
            prompt=prompt,
            requested_tokens=requested_tokens,
            model_id=model
        )

        # Reserve in ledger
        reserved = self.ledger.reserve(task_id, agent, model, effective_tokens)

        if not reserved:
            return False, 0, "Insufficient global budget"

        return True, effective_tokens, "approved"

    def activate_allocation(self, task_id: str) -> None:
        """Mark allocation as active."""
        self.ledger.activate(task_id)

    def release_allocation(self, task_id: str, success: bool = True) -> None:
        """Release allocated tokens."""
        status = "completed" if success else "failed"
        self.ledger.release(task_id, status)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.ledger.get_usage_stats()

    def is_emergency_shutdown(self) -> bool:
        """Check if emergency shutdown was triggered."""
        return self._emergency_shutdown


# =============================================================================
# RESOURCE ANALYZER
# =============================================================================

class ResourceAnalyzer:
    """
    The Aperture of the Singularity.

    Manages tiered data access and dynamic context windowing.
    Automatically throttles ingestion based on hardware telemetry.
    """

    VRAM_CLIFF_MB = 9750.0  # The 9.75 GB Cliff
    SAFETY_MARGIN = 0.85    # 85% Safety Threshold

    def __init__(self):
        self.active_shard = "general"
        self.context_window_size = 4096

    def get_vram_usage(self) -> float:
        """Get current VRAM usage in MB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return 0.0

    def analyze_pressure(self) -> Dict[str, Any]:
        """Calculate the current cognitive pressure."""
        current_vram = self.get_vram_usage()
        utilization = current_vram / self.VRAM_CLIFF_MB if self.VRAM_CLIFF_MB > 0 else 0

        status = "HEALTHY"
        if utilization > 0.95:
            status = "CRITICAL_COLLAPSE"
        elif utilization > self.SAFETY_MARGIN:
            status = "WINCE_ALERT"

        return {
            "vram_mb": current_vram,
            "utilization": utilization,
            "status": status,
            "can_ingest": status == "HEALTHY"
        }

    def adjust_aperture(self, data_size_mb: float) -> None:
        """Dynamically adjust the context window based on pressure."""
        pressure = self.analyze_pressure()

        if pressure["status"] != "HEALTHY":
            self.context_window_size = max(512, self.context_window_size // 2)
            logger.warning(f"[APERTURE] Pressure detected. Shrinking context to {self.context_window_size}")
        else:
            self.context_window_size = min(8192, self.context_window_size + 512)
            logger.info(f"[APERTURE] Resource headroom available. Context window: {self.context_window_size}")

    def request_shard(self, shard_name: str) -> bool:
        """
        Tiered Access Logic: NVMe -> NPU -> GPU.

        Args:
            shard_name: Name of the shard to load

        Returns:
            True if shard can be loaded
        """
        if shard_name == self.active_shard:
            return True

        pressure = self.analyze_pressure()
        if not pressure["can_ingest"]:
            logger.error(f"[TIERED_ACCESS] Denied load of shard '{shard_name}'. VRAM near Cliff.")
            return False

        logger.info(f"[TIERED_ACCESS] Swapping shard: {self.active_shard} -> {shard_name}")
        self.active_shard = shard_name
        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_budget(
    prompt: str,
    requested_tokens: int,
    model_id: Optional[str] = None,
    profile: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Compute effective token budget."""
    orchestrator = BudgetOrchestrator()
    return orchestrator.compute(prompt, requested_tokens, model_id, profile)


def apply_dynamic_budget(
    prompt: str,
    requested_tokens: int,
    model_id: Optional[str] = None,
    profile: Optional[str] = None,
) -> int:
    """Apply dynamic budget and return effective tokens."""
    effective, _ = compute_budget(prompt, requested_tokens, model_id, profile)
    return effective


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_budget_ledger: Optional[BudgetLedger] = None
_budget_service: Optional[BudgetService] = None
_ledger_lock = threading.Lock()
_service_lock = threading.Lock()


def create_budget_ledger() -> BudgetLedger:
    """Create a new BudgetLedger instance."""
    return BudgetLedger()


def get_budget_ledger() -> BudgetLedger:
    """Get or create the global budget ledger."""
    global _budget_ledger
    if _budget_ledger is None:
        with _ledger_lock:
            if _budget_ledger is None:
                _budget_ledger = BudgetLedger()
    return _budget_ledger


def create_budget_service() -> BudgetService:
    """Create a new BudgetService instance."""
    return BudgetService()


def get_budget_service() -> BudgetService:
    """Get or create the global budget service."""
    global _budget_service
    if _budget_service is None:
        with _service_lock:
            if _budget_service is None:
                _budget_service = BudgetService()
    return _budget_service


__all__ = [
    # Ledger
    "BudgetLedger", "BudgetAllocation", "get_budget_ledger",
    # Orchestrator
    "BudgetOrchestrator", "BudgetSettings", "compute_budget", "apply_dynamic_budget",
    # Service
    "BudgetService", "ResourceSnapshot", "SafetyLimits", "get_budget_service",
    # Analyzer
    "ResourceAnalyzer",
    # Factory
    "create_budget_ledger", "create_budget_service",
]
