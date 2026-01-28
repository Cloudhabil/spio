"""
ASIOS - Autonomous Operating System Runtime

The runtime layer of Sovereign PIO.
Handles governance, resource management, failsafe, and execution.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any

from sovereign_pio.constants import PHI, BETA

# Import full implementations
from .governor import Governor, HardwareMonitor, GPUStatus, SystemStatus

__all__ = [
    "ASIOSRuntime",
    "FailSafe",
    "Governor",
    "HardwareMonitor",
    "GPUStatus",
    "SystemStatus",
]


@dataclass
class FailSafe:
    """
    Circuit breaker for failure isolation.

    Implements sliding window failure tracking with automatic cooldown.
    """

    threshold: int = 3
    window_seconds: float = 30.0
    cooldown_seconds: float = 60.0

    _failures: list = field(default_factory=list)
    _tripped_at: float = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_failure(self):
        """Record a failure event."""
        with self._lock:
            now = time.time()
            self._failures.append(now)
            cutoff = now - self.window_seconds
            self._failures = [t for t in self._failures if t > cutoff]

            if len(self._failures) >= self.threshold:
                self._tripped_at = now

    def record_success(self):
        """Record a success event, reset failure count."""
        with self._lock:
            self._failures = []

    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped."""
        with self._lock:
            if self._tripped_at is None:
                return False
            if time.time() - self._tripped_at > self.cooldown_seconds:
                self._tripped_at = None
                self._failures = []
                return False
            return True

    def execute(self, func: Callable, fallback: Callable = None):
        """Execute function with circuit breaker protection."""
        if self.is_tripped():
            if fallback:
                return fallback()
            raise RuntimeError("Circuit breaker is tripped")

        try:
            result = func()
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            if fallback and self.is_tripped():
                return fallback()
            raise


class ASIOSRuntime:
    """
    ASIOS Runtime Engine.

    Manages execution, governance, and resource allocation
    for Sovereign PIO operations.
    """

    def __init__(self):
        self.failsafe = FailSafe()
        self.governor = Governor()
        self.phi = PHI

    async def execute(self, task: dict) -> dict:
        """Execute a task through the runtime."""
        # Check system health
        health = self.governor.check_health()

        if health["critical"]:
            return {"status": "error", "reason": "Critical stop active"}

        if health["throttled"]:
            return {"status": "throttled", "reason": "System throttled"}

        # Execute with failsafe protection
        def run_task():
            return {"status": "success", "result": task}

        try:
            return self.failsafe.execute(run_task)
        except RuntimeError as e:
            return {"status": "error", "reason": str(e)}

    def get_status(self) -> dict:
        """Get current runtime status."""
        return {
            "failsafe_tripped": self.failsafe.is_tripped(),
            "throttled": self.governor.is_throttled,
            "critical_stop": self.governor.critical_stop,
            "governor": self.governor.check_health(),
        }

    def summary(self) -> str:
        """Get human-readable status summary."""
        return self.governor.summary()
