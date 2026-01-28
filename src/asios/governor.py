"""
ASIOS Resource Governor with Hardware Integration

Monitors and controls hardware resources (GPU, CPU, memory, disk)
with real nvidia-smi integration for VRAM management.
"""

import subprocess
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Import from parent package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sovereign_pio.constants import BETA


@dataclass
class GPUStatus:
    """Current GPU status from nvidia-smi."""
    name: str = ""
    vram_total_mb: float = 0
    vram_used_mb: float = 0
    vram_free_mb: float = 0
    temperature_c: float = 0
    utilization_percent: float = 0
    power_watts: float = 0
    available: bool = False
    error: Optional[str] = None


@dataclass
class SystemStatus:
    """Overall system resource status."""
    gpu: GPUStatus = field(default_factory=GPUStatus)
    cpu_percent: float = 0
    ram_used_mb: float = 0
    ram_total_mb: float = 0
    disk_free_gb: float = 0


class HardwareMonitor:
    """
    Hardware monitoring with nvidia-smi integration.

    Provides real-time GPU metrics for resource governance.
    """

    def __init__(self):
        self._nvidia_smi_path = self._find_nvidia_smi()

    def _find_nvidia_smi(self) -> Optional[str]:
        """Find nvidia-smi executable."""
        # Common paths
        paths = [
            "nvidia-smi",  # In PATH
            "C:/Windows/System32/nvidia-smi.exe",
            "C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe",
            "/usr/bin/nvidia-smi",
            "/usr/local/bin/nvidia-smi",
        ]

        for path in paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (subprocess.SubprocessError, FileNotFoundError, OSError):
                continue

        return None

    def query_gpu(self) -> GPUStatus:
        """Query GPU status using nvidia-smi."""
        status = GPUStatus()

        if not self._nvidia_smi_path:
            status.error = "nvidia-smi not found"
            return status

        try:
            # Query GPU metrics
            result = subprocess.run(
                [
                    self._nvidia_smi_path,
                    "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                status.error = f"nvidia-smi failed: {result.stderr}"
                return status

            # Parse CSV output
            line = result.stdout.strip().split('\n')[0]  # First GPU
            parts = [p.strip() for p in line.split(',')]

            if len(parts) >= 7:
                status.name = parts[0]
                status.vram_total_mb = float(parts[1])
                status.vram_used_mb = float(parts[2])
                status.vram_free_mb = float(parts[3])
                status.temperature_c = float(parts[4])
                status.utilization_percent = float(parts[5])
                # Power might be [N/A] for some GPUs
                try:
                    status.power_watts = float(parts[6])
                except ValueError:
                    status.power_watts = 0
                status.available = True

        except subprocess.TimeoutExpired:
            status.error = "nvidia-smi timeout"
        except Exception as e:
            status.error = str(e)

        return status

    def get_system_status(self) -> SystemStatus:
        """Get complete system status."""
        status = SystemStatus()
        status.gpu = self.query_gpu()

        # CPU and RAM (cross-platform)
        try:
            import psutil
            status.cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            status.ram_used_mb = mem.used / (1024 * 1024)
            status.ram_total_mb = mem.total / (1024 * 1024)
            disk = psutil.disk_usage('/')
            status.disk_free_gb = disk.free / (1024 * 1024 * 1024)
        except ImportError:
            # psutil not available, skip
            pass

        return status


class Governor:
    """
    Resource Governor with PHI-based safety limits.

    Implements hardware protection with real monitoring:
    - VRAM cliff protection (81.25% limit for Windows DWM stability)
    - Thermal throttling
    - Disk space monitoring
    """

    def __init__(
        self,
        vram_limit_percent: float = 81.25,  # VRAM cliff for Windows
        temp_limit_c: float = 78.0,          # Thermal throttle point
        disk_min_gb: float = 50.0,           # Minimum free disk
    ):
        self.vram_limit_percent = vram_limit_percent
        self.temp_limit_c = temp_limit_c
        self.disk_min_gb = disk_min_gb
        self.safety_factor = BETA  # 0.236 security constant

        self.monitor = HardwareMonitor()

        # State
        self.is_throttled = False
        self.critical_stop = False
        self._last_status: Optional[SystemStatus] = None

    @property
    def vram_limit_mb(self) -> float:
        """Dynamic VRAM limit based on actual GPU."""
        status = self.get_status()
        if status.gpu.available:
            return status.gpu.vram_total_mb * (self.vram_limit_percent / 100)
        return 9750  # Default for 12GB GPU

    def get_status(self) -> SystemStatus:
        """Get current system status."""
        self._last_status = self.monitor.get_system_status()
        return self._last_status

    def check_health(self) -> dict:
        """
        Check system health against all limits.

        Returns:
            Dict with health status and any warnings/errors
        """
        status = self.get_status()
        health = {
            "healthy": True,
            "throttled": False,
            "critical": False,
            "warnings": [],
            "errors": [],
        }

        if not status.gpu.available:
            health["warnings"].append(f"GPU not available: {status.gpu.error}")
        else:
            # VRAM check
            vram_percent = (status.gpu.vram_used_mb / status.gpu.vram_total_mb) * 100
            if vram_percent >= self.vram_limit_percent:
                health["throttled"] = True
                health["warnings"].append(
                    f"VRAM at {vram_percent:.1f}% (limit: {self.vram_limit_percent}%)"
                )

            # Temperature check
            if status.gpu.temperature_c >= self.temp_limit_c:
                health["throttled"] = True
                health["warnings"].append(
                    f"GPU temp at {status.gpu.temperature_c}C (limit: {self.temp_limit_c}C)"
                )

            # Critical temperature
            if status.gpu.temperature_c >= 85:
                health["critical"] = True
                health["errors"].append(
                    f"CRITICAL: GPU temp at {status.gpu.temperature_c}C"
                )

        # Disk check
        if status.disk_free_gb > 0 and status.disk_free_gb < self.disk_min_gb:
            health["warnings"].append(
                f"Low disk space: {status.disk_free_gb:.1f}GB free"
            )

        health["healthy"] = not (health["throttled"] or health["critical"])
        self.is_throttled = health["throttled"]
        self.critical_stop = health["critical"]

        return health

    def check_vram(self, current_mb: float = None) -> bool:
        """Check if VRAM usage is within limits."""
        if current_mb is None:
            status = self.get_status()
            if not status.gpu.available:
                return True  # Can't check, allow
            current_mb = status.gpu.vram_used_mb

        limit = self.vram_limit_mb
        if current_mb >= limit:
            self.is_throttled = True
            return False
        return True

    def check_temperature(self, current_c: float = None) -> bool:
        """Check if temperature is within limits."""
        if current_c is None:
            status = self.get_status()
            if not status.gpu.available:
                return True
            current_c = status.gpu.temperature_c

        if current_c >= self.temp_limit_c:
            self.is_throttled = True
            return False
        return True

    def get_safe_allocation(self, requested_mb: float) -> float:
        """
        Get safe VRAM allocation with safety buffer.

        Uses BETA (security constant) as safety margin.
        """
        status = self.get_status()

        if not status.gpu.available:
            # Conservative estimate for unknown GPU
            max_safe = 8000 * (1 - self.safety_factor)
            return min(requested_mb, max_safe)

        available = status.gpu.vram_free_mb
        # Apply safety factor
        safe_available = available * (1 - self.safety_factor)

        return min(requested_mb, safe_available)

    def can_allocate(self, requested_mb: float) -> bool:
        """Check if requested VRAM can be safely allocated."""
        return self.get_safe_allocation(requested_mb) >= requested_mb

    def summary(self) -> str:
        """Get human-readable status summary."""
        status = self.get_status()
        health = self.check_health()

        lines = ["=== Governor Status ==="]

        if status.gpu.available:
            lines.append(f"GPU: {status.gpu.name}")
            lines.append(f"VRAM: {status.gpu.vram_used_mb:.0f}/{status.gpu.vram_total_mb:.0f} MB "
                        f"({status.gpu.vram_used_mb/status.gpu.vram_total_mb*100:.1f}%)")
            lines.append(f"Temp: {status.gpu.temperature_c}C")
            lines.append(f"Util: {status.gpu.utilization_percent}%")
        else:
            lines.append(f"GPU: Not available ({status.gpu.error})")

        lines.append(f"Throttled: {self.is_throttled}")
        lines.append(f"Critical: {self.critical_stop}")

        if health["warnings"]:
            lines.append("Warnings: " + "; ".join(health["warnings"]))

        return "\n".join(lines)
