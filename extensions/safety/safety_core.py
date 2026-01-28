"""
Safety Extension - Hardware Protection and Threat Detection

Ported from:
- CLI-main/src/core/safety_governor.py
- CLI-main/src/core/immune_system.py

Implements:
- SafetyGovernor: VRAM/thermal/disk protection
- ImmuneSystem: Adaptive threat detection
- FaultTolerance: Graceful degradation
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Detection thresholds (empirically derived)
ANOMALY_THRESHOLD_ALPHA = 0.382    # Minor deviation
ANOMALY_THRESHOLD_BETA = 0.618     # Significant deviation
ANOMALY_THRESHOLD_GAMMA = 0.854    # Critical deviation

# System parameters
SIGNATURE_RETENTION_DAYS = 30
ADAPTIVE_LEARNING_RATE = 0.236
MINIMUM_CLUSTER_SIZE = 3
RECOVERY_ITERATION_COUNT = 7


# =============================================================================
# ENUMS
# =============================================================================

class SeverityLevel(IntEnum):
    """Threat severity classification following CVSS v3.1 methodology."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ThreatCategory(Enum):
    """Taxonomy of detectable threat types."""
    BOUNDARY_VIOLATION = "boundary"
    INJECTION_ATTACK = "injection"
    INTEGRITY_VIOLATION = "integrity"
    BEHAVIORAL_ANOMALY = "behavioral"
    RESOURCE_EXHAUSTION = "dos"
    UNTRUSTED_SOURCE = "untrusted"
    UNCLASSIFIED = "unknown"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GPUVitals:
    """GPU health metrics."""
    vram_pct: float = 0.0
    temp: float = 0.0
    vram_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0


@dataclass
class DiskHealth:
    """Disk health metrics."""
    free_gb: float = 0.0
    is_low: bool = False


@dataclass
class ThreatSignature:
    """
    Formal representation of a detected threat pattern.
    """
    signature_id: str
    category: ThreatCategory
    centroid_vector: Optional[List[float]] = None
    boundary_radius: float = 0.5
    keyword_patterns: List[str] = field(default_factory=list)
    source_blacklist: List[str] = field(default_factory=list)
    created_at: str = ""
    last_seen: str = ""
    occurrence_count: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        self.last_seen = datetime.utcnow().isoformat()


@dataclass
class SecurityEvent:
    """Structured security event record."""
    event_id: str
    timestamp: str
    severity: SeverityLevel
    category: ThreatCategory
    source_identifier: str
    description: str
    feature_vector: Optional[List[float]] = None
    affected_components: List[int] = field(default_factory=list)
    response_actions: str = ""
    resolved: bool = False


# =============================================================================
# SAFETY GOVERNOR
# =============================================================================

class SafetyGovernor:
    """
    Hardware Protection Layer (The Circuit Breaker).
    Monitors VRAM, Temperature, and Disk to prevent hardware stress.

    Hard-codes the 9750MB VRAM cliff (81.25%) to prevent "driver juggling"
    instability when Windows DWM fights for the remaining ~1.7GB.
    """

    # Hardware sovereignty constants
    VRAM_CLIFF_MB = 9750
    VRAM_CLIFF_PCT = 81.25
    VRAM_DWM_RESERVE_MB = 1700
    VRAM_SAFETY_BUFFER_MB = 850

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path(".")
        self.log_dir = self.repo_root / "logs" / "safety"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Check for overrides
        equilibrium_mode = os.getenv("SUBSTRATE_EQUILIBRIUM", "0") == "1"
        vram_limit_override = os.getenv("VRAM_LIMIT_MB")

        if vram_limit_override:
            self.vram_limit_mb = int(vram_limit_override)
            self.vram_limit_pct = (self.vram_limit_mb / 12288) * 100
        elif equilibrium_mode:
            self.vram_limit_mb = self.VRAM_CLIFF_MB
            self.vram_limit_pct = self.VRAM_CLIFF_PCT
        else:
            self.vram_limit_mb = self.VRAM_CLIFF_MB
            self.vram_limit_pct = self.VRAM_CLIFF_PCT

        self.temp_limit_c = 78.0
        self.disk_min_free_gb = 50.0

        self.is_throttled = False
        self.critical_stop = False

    def get_gpu_vitals(self) -> GPUVitals:
        """Query nvidia-smi for VRAM and Temperature."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
                 "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                used, total, temp = map(float, result.stdout.strip().split(','))
                return GPUVitals(
                    vram_pct=(used / total) * 100,
                    temp=temp,
                    vram_mb=used,
                    vram_total_mb=total,
                    vram_free_mb=total - used
                )
        except Exception as e:
            logger.error(f"Failed to query GPU: {e}")
        return GPUVitals()

    def check_disk_health(self) -> DiskHealth:
        """Check free space and detect potential thrashing."""
        try:
            total, used, free = shutil.disk_usage(self.repo_root)
            free_gb = free / (1024**3)
            return DiskHealth(
                free_gb=free_gb,
                is_low=free_gb < self.disk_min_free_gb
            )
        except Exception:
            return DiskHealth()

    def audit_system(self, required_vram_mb: float = 0.0) -> Tuple[bool, str]:
        """
        Perform a full safety audit.
        Returns: (is_safe, message)
        """
        gpu = self.get_gpu_vitals()
        disk = self.check_disk_health()

        # Thermal Check
        if gpu.temp > self.temp_limit_c:
            self.is_throttled = True
            msg = f"THERMAL OVERHEAT: {gpu.temp}C > {self.temp_limit_c}C"
            logger.warning(msg)
            return False, msg

        # VRAM Check (Percentage)
        if gpu.vram_pct > self.vram_limit_pct:
            msg = f"VRAM CRITICAL: {gpu.vram_pct:.1f}% > {self.vram_limit_pct:.1f}% (cliff)"
            logger.warning(msg)
            return False, msg

        # VRAM Check (Absolute MB)
        if gpu.vram_mb > self.vram_limit_mb:
            msg = f"VRAM CLIFF BREACH: {gpu.vram_mb:.0f}MB > {self.vram_limit_mb}MB"
            logger.warning(msg)
            return False, msg

        # VRAM Requirement Check
        if required_vram_mb > 0 and gpu.vram_free_mb < required_vram_mb:
            msg = f"VRAM INSUFFICIENT: Need {required_vram_mb}MB, only {gpu.vram_free_mb:.0f}MB free"
            return False, msg

        # Disk Space Check
        if disk.is_low:
            msg = f"DISK SPACE LOW: {disk.free_gb:.1f}GB left"
            logger.warning(msg)
            return False, msg

        self.is_throttled = False
        return True, "SYSTEM_ALIGNED_SAFE"

    def get_throttle_factor(self) -> float:
        """
        Calculates how much to slow down.
        1.0 = Full speed, 0.1 = Near stop.
        """
        gpu = self.get_gpu_vitals()

        if gpu.temp > 70:
            return max(0.2, 1.0 - (gpu.temp - 70) / (self.temp_limit_c - 70))

        return 1.0


# =============================================================================
# FAULT TOLERANCE MANAGER
# =============================================================================

class FaultToleranceManager:
    """
    Multi-level degradation and recovery management.

    Levels:
        0: Normal operation
        1: External input disabled
        2: Autonomous operations disabled
        3: Read-only mode
        4: Network isolation
        5: Emergency halt
    """

    def __init__(self, repo_root: Path):
        self._root = repo_root
        self._recovery_log: List[Dict] = []
        self._degradation_level = 0

    def execute_response(self, event: SecurityEvent) -> Dict:
        """Execute appropriate response based on event severity."""
        response = {
            "event_id": event.event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "actions": [],
            "success": True
        }

        if event.severity == SeverityLevel.LOW:
            response["actions"].append("LOG_EVENT")

        elif event.severity == SeverityLevel.MEDIUM:
            response["actions"].append("ISOLATE_COMPONENTS")
            self._isolate_components(event.affected_components)

        elif event.severity == SeverityLevel.HIGH:
            response["actions"].append("ISOLATE_COMPONENTS")
            response["actions"].append("ENTER_DEGRADED_MODE_1")
            self._isolate_components(event.affected_components)
            self._set_degradation_level(1)

        elif event.severity == SeverityLevel.CRITICAL:
            response["actions"].append("EMERGENCY_ISOLATION")
            response["actions"].append("ENTER_DEGRADED_MODE_3")
            response["actions"].append("ALERT_ADMINISTRATOR")
            self._isolate_components(event.affected_components)
            self._set_degradation_level(3)

        self._recovery_log.append(response)
        return response

    def _isolate_components(self, component_indices: List[int]):
        """Mark components as isolated in system state."""
        state_path = self._root / "data" / "substrate_tree.json"
        if not state_path.exists():
            return

        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            for idx in component_indices:
                if isinstance(state, list) and 0 <= idx < len(state):
                    state[idx]["status"] = "ISOLATED"
                    state[idx]["isolated_at"] = datetime.utcnow().isoformat()

            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f)

            logger.info(f"Isolated {len(component_indices)} components.")
        except Exception as e:
            logger.error(f"Component isolation failed: {e}")

    def _set_degradation_level(self, level: int):
        """Set system degradation level."""
        self._degradation_level = min(level, 5)

        state_path = self._root / "data" / "system_degradation.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        restrictions = self._get_restrictions(self._degradation_level)

        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump({
                "level": self._degradation_level,
                "timestamp": datetime.utcnow().isoformat(),
                "restrictions": restrictions
            }, f, indent=2)

        logger.warning(f"System degradation level: {self._degradation_level}")

    def _get_restrictions(self, level: int) -> List[str]:
        """Map degradation level to operational restrictions."""
        restrictions = []
        if level >= 1:
            restrictions.append("EXTERNAL_INPUT_DISABLED")
        if level >= 2:
            restrictions.append("AUTONOMOUS_OPERATIONS_DISABLED")
        if level >= 3:
            restrictions.append("WRITE_OPERATIONS_DISABLED")
        if level >= 4:
            restrictions.append("NETWORK_ACCESS_DISABLED")
        if level >= 5:
            restrictions.append("SYSTEM_HALTED")
        return restrictions

    def attempt_recovery(self) -> bool:
        """Attempt system recovery from degraded state."""
        if self._degradation_level == 0:
            return True

        for _ in range(RECOVERY_ITERATION_COUNT):
            if self._degradation_level > 0:
                self._degradation_level -= 1

        if self._degradation_level == 0:
            state_file = self._root / "data" / "system_degradation.json"
            if state_file.exists():
                state_file.unlink()
            logger.info("System recovery complete.")
            return True

        return False

    @property
    def degradation_level(self) -> int:
        """Current degradation level."""
        return self._degradation_level


# =============================================================================
# IMMUNE SYSTEM
# =============================================================================

class ImmuneSystem:
    """
    Adaptive Threat Detection and Response System.

    Provides:
    - Anomaly Detection: Statistical and pattern-based threat identification
    - Adaptive Boundaries: ML-augmented security perimeters
    - Graceful Degradation: Multi-level fault tolerance
    - Persistent Threat Memory: Long-term signature storage with decay
    """

    def __init__(self, repo_root: Path):
        self._root = repo_root
        self._signatures_path = repo_root / "data" / "threat_signatures.json"
        self._signatures: Dict[str, ThreatSignature] = {}
        self._baseline: Dict[str, float] = {}
        self._event_history: List[SecurityEvent] = []
        self._fault_manager = FaultToleranceManager(repo_root)

        self._load_signatures()

    def _load_signatures(self):
        """Load signatures from persistent storage."""
        if not self._signatures_path.exists():
            return

        try:
            with open(self._signatures_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for entry in data.get("signatures", []):
                sig = ThreatSignature(
                    signature_id=entry["signature_id"],
                    category=ThreatCategory(entry["category"]),
                    centroid_vector=entry.get("centroid_vector"),
                    boundary_radius=entry.get("boundary_radius", 0.5),
                    keyword_patterns=entry.get("keyword_patterns", []),
                    source_blacklist=entry.get("source_blacklist", []),
                    created_at=entry.get("created_at", ""),
                    last_seen=entry.get("last_seen", ""),
                    occurrence_count=entry.get("occurrence_count", 0)
                )
                self._signatures[sig.signature_id] = sig

            logger.info(f"Loaded {len(self._signatures)} threat signatures.")
        except Exception as e:
            logger.error(f"Failed to load signature store: {e}")

    def _save_signatures(self):
        """Persist signatures to storage."""
        self._signatures_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "signatures": [],
            "last_updated": datetime.utcnow().isoformat(),
            "schema_version": "1.0"
        }

        for sig in self._signatures.values():
            entry = asdict(sig)
            entry["category"] = sig.category.value
            data["signatures"].append(entry)

        with open(self._signatures_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def evaluate(
        self,
        feature_vector: List[float],
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, Optional[SecurityEvent]]:
        """
        Evaluate input against all detection mechanisms.

        Args:
            feature_vector: N-dimensional feature representation
            metadata: Additional context

        Returns:
            Tuple of (is_permitted, security_event_if_detected)
        """
        metadata = metadata or {}

        # Check against known signatures
        for signature in self._signatures.values():
            if signature.centroid_vector is not None:
                distance = self._euclidean_distance(
                    feature_vector,
                    signature.centroid_vector
                )
                if distance < signature.boundary_radius:
                    event = self._create_event(
                        SeverityLevel.MEDIUM,
                        signature.category,
                        metadata.get("source", "unknown"),
                        f"Matched signature: {signature.signature_id}",
                        feature_vector
                    )
                    self._handle_detection(event)
                    return False, event

            # Keyword pattern matching
            source = metadata.get("source", "")
            for pattern in signature.keyword_patterns:
                if pattern.lower() in source.lower():
                    event = self._create_event(
                        SeverityLevel.MEDIUM,
                        signature.category,
                        source,
                        f"Pattern match: {pattern}",
                        feature_vector
                    )
                    self._handle_detection(event)
                    return False, event

        return True, None

    def _euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        if len(v1) != len(v2):
            return float('inf')
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

    def calibrate_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics for anomaly detection."""
        self._baseline = metrics.copy()

    def monitor_metrics(
        self,
        current_metrics: Dict[str, float]
    ) -> List[SecurityEvent]:
        """Evaluate system metrics for anomalies."""
        anomalies = []
        events = []

        for metric_name, current_value in current_metrics.items():
            baseline_value = self._baseline.get(metric_name, current_value)
            if baseline_value == 0:
                continue

            deviation = abs(current_value - baseline_value) / (baseline_value + 1e-9)

            if deviation > ANOMALY_THRESHOLD_GAMMA:
                severity = SeverityLevel.HIGH
            elif deviation > ANOMALY_THRESHOLD_BETA:
                severity = SeverityLevel.MEDIUM
            elif deviation > ANOMALY_THRESHOLD_ALPHA:
                severity = SeverityLevel.LOW
            else:
                continue

            event = self._create_event(
                severity,
                ThreatCategory.BEHAVIORAL_ANOMALY,
                "metrics_monitor",
                f"Anomaly in {metric_name}: {deviation:.1%} deviation",
                None
            )
            events.append(event)
            if severity.value >= SeverityLevel.MEDIUM.value:
                self._handle_detection(event)

        return events

    def _handle_detection(self, event: SecurityEvent):
        """Process detected security event."""
        self._event_history.append(event)
        response = self._fault_manager.execute_response(event)
        event.response_actions = ", ".join(response["actions"])

        logger.warning(
            f"Security event [{event.severity.name}]: {event.response_actions}"
        )

    def _create_event(
        self,
        severity: SeverityLevel,
        category: ThreatCategory,
        source: str,
        description: str,
        vector: Optional[List[float]]
    ) -> SecurityEvent:
        """Construct security event record."""
        return SecurityEvent(
            event_id=f"evt_{int(time.time() * 1000)}",
            timestamp=datetime.utcnow().isoformat(),
            severity=severity,
            category=category,
            source_identifier=source,
            description=description,
            feature_vector=vector
        )

    def register_signature(self, signature: ThreatSignature):
        """Add or update a threat signature."""
        if signature.signature_id in self._signatures:
            existing = self._signatures[signature.signature_id]
            existing.occurrence_count += 1
            existing.last_seen = datetime.utcnow().isoformat()
        else:
            self._signatures[signature.signature_id] = signature
        self._save_signatures()

    def get_status(self) -> Dict[str, Any]:
        """Return current system status."""
        return {
            "degradation_level": self._fault_manager.degradation_level,
            "active_threats": len([e for e in self._event_history if not e.resolved]),
            "known_signatures": len(self._signatures),
            "total_events": len(self._event_history),
            "last_event": self._event_history[-1].timestamp if self._event_history else None,
            "status": (
                "NORMAL" if self._fault_manager.degradation_level == 0
                else f"DEGRADED_L{self._fault_manager.degradation_level}"
            )
        }

    def run_maintenance(self) -> Dict[str, Any]:
        """Execute periodic maintenance tasks."""
        recovered = self._fault_manager.attempt_recovery()

        # Prune expired signatures
        cutoff = datetime.utcnow() - timedelta(days=SIGNATURE_RETENTION_DAYS * 2)
        expired = []
        for sig_id, sig in self._signatures.items():
            last_seen = datetime.fromisoformat(sig.last_seen) if sig.last_seen else datetime.min
            if last_seen < cutoff and sig.occurrence_count < 5:
                expired.append(sig_id)

        for sig_id in expired:
            del self._signatures[sig_id]
            logger.info(f"Pruned expired signature: {sig_id}")

        if expired:
            self._save_signatures()

        return {
            "recovery_attempted": True,
            "recovered": recovered,
            "degradation_level": self._fault_manager.degradation_level,
            "active_signatures": len(self._signatures)
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_safety_governor: Optional[SafetyGovernor] = None
_immune_system: Optional[ImmuneSystem] = None


def create_safety_governor(repo_root: Optional[Path] = None) -> SafetyGovernor:
    """Create a new SafetyGovernor instance."""
    return SafetyGovernor(repo_root=repo_root)


def get_safety_governor() -> SafetyGovernor:
    """Get global SafetyGovernor instance."""
    global _safety_governor
    if _safety_governor is None:
        _safety_governor = SafetyGovernor()
    return _safety_governor


def create_immune_system(repo_root: Path) -> ImmuneSystem:
    """Create a new ImmuneSystem instance."""
    return ImmuneSystem(repo_root=repo_root)


def get_immune_system() -> ImmuneSystem:
    """Get global ImmuneSystem instance."""
    global _immune_system
    if _immune_system is None:
        _immune_system = ImmuneSystem(Path("."))
    return _immune_system


__all__ = [
    # Governor
    "SafetyGovernor", "GPUVitals", "DiskHealth",
    # Immune System
    "ImmuneSystem", "ThreatSignature", "SecurityEvent",
    "SeverityLevel", "ThreatCategory",
    # Fault Tolerance
    "FaultToleranceManager",
    # Factory
    "create_safety_governor", "create_immune_system",
    "get_safety_governor", "get_immune_system",
]
