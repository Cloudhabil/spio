"""
Observability - Structured Logging and Distributed Tracing

Provides unified JSON-based logging with trace context propagation
for debugging autonomous agent systems.

Features:
- JSONL output for machine consumption
- Distributed trace context (trace_id, span_id, parent_id)
- Custom metrics with tags
- Alert level classification
"""

import json
import time
import uuid
import sys
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TextIO
from enum import Enum
from pathlib import Path
from contextvars import ContextVar


class AlertLevel(Enum):
    """Alert severity classification."""
    NONE = "none"
    INFO = "info"
    INTERFACE_GLITCH = "interface_glitch"
    DEGRADED = "degraded"
    INFRASTRUCTURE_WARNING = "infrastructure_warning"
    INFRASTRUCTURE_CRITICAL = "infrastructure_critical"
    SECURITY = "security"


class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TraceContext:
    """Distributed trace context."""
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None

    @classmethod
    def new(cls) -> "TraceContext":
        """Create a new trace context."""
        return cls(
            trace_id=uuid.uuid4().hex[:16],
            span_id=uuid.uuid4().hex[:8],
        )

    def child(self) -> "TraceContext":
        """Create a child span context."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:8],
            parent_id=self.span_id,
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
        }


# Context variable for current trace
_current_trace: ContextVar[Optional[TraceContext]] = ContextVar(
    "current_trace", default=None
)


def get_current_trace() -> Optional[TraceContext]:
    """Get the current trace context."""
    return _current_trace.get()


def set_current_trace(ctx: TraceContext) -> None:
    """Set the current trace context."""
    _current_trace.set(ctx)


@dataclass
class LogEntry:
    """A structured log entry."""
    timestamp: float
    level: LogLevel
    message: str
    logger: str
    trace: Optional[TraceContext] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    alert: AlertLevel = AlertLevel.NONE

    def to_dict(self) -> dict:
        entry = {
            "ts": self.timestamp,
            "level": self.level.value,
            "msg": self.message,
            "logger": self.logger,
        }

        if self.trace:
            entry["trace"] = self.trace.to_dict()

        if self.tags:
            entry["tags"] = self.tags

        if self.metrics:
            entry["metrics"] = self.metrics

        if self.alert != AlertLevel.NONE:
            entry["alert"] = self.alert.value

        return entry

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class StructuredLogger:
    """
    Structured logger with JSON output and trace context.

    Example:
        logger = StructuredLogger("pio.operator")

        with logger.span("process_request") as span:
            logger.info("Processing request", tags={"user": "alice"})
            logger.metric("latency_ms", 42.5)
    """

    def __init__(
        self,
        name: str,
        output: TextIO = None,
        min_level: LogLevel = LogLevel.DEBUG,
    ):
        self.name = name
        self.output = output or sys.stderr
        self.min_level = min_level
        self._lock = threading.Lock()

    def _should_log(self, level: LogLevel) -> bool:
        """Check if level should be logged."""
        levels = list(LogLevel)
        return levels.index(level) >= levels.index(self.min_level)

    def _emit(self, entry: LogEntry):
        """Emit a log entry."""
        if not self._should_log(entry.level):
            return

        line = entry.to_json() + "\n"
        with self._lock:
            self.output.write(line)
            self.output.flush()

    def log(
        self,
        level: LogLevel,
        message: str,
        tags: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        alert: AlertLevel = AlertLevel.NONE,
    ):
        """Log a message at the specified level."""
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            logger=self.name,
            trace=get_current_trace(),
            tags=tags or {},
            metrics=metrics or {},
            alert=alert,
        )
        self._emit(entry)

    def debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def metric(self, name: str, value: float, tags: Optional[Dict[str, Any]] = None):
        """Log a metric value."""
        self.log(
            LogLevel.INFO,
            f"metric:{name}",
            tags=tags,
            metrics={name: value},
        )

    def alert(self, level: AlertLevel, message: str, tags: Optional[Dict[str, Any]] = None):
        """Log an alert."""
        log_level = LogLevel.WARNING
        if level in (AlertLevel.INFRASTRUCTURE_CRITICAL, AlertLevel.SECURITY):
            log_level = LogLevel.CRITICAL
        elif level == AlertLevel.INFRASTRUCTURE_WARNING:
            log_level = LogLevel.WARNING

        self.log(log_level, message, tags=tags, alert=level)

    def span(self, name: str) -> "SpanContext":
        """Create a new span context."""
        return SpanContext(self, name)

    def child(self, name: str) -> "StructuredLogger":
        """Create a child logger."""
        return StructuredLogger(
            f"{self.name}.{name}",
            output=self.output,
            min_level=self.min_level,
        )


class SpanContext:
    """
    Context manager for tracing spans.

    Automatically tracks duration and propagates trace context.
    """

    def __init__(self, logger: StructuredLogger, name: str):
        self.logger = logger
        self.name = name
        self.start_time: float = 0
        self.trace: Optional[TraceContext] = None
        self._token = None

    def __enter__(self) -> "SpanContext":
        self.start_time = time.time()

        # Create or inherit trace context
        parent = get_current_trace()
        if parent:
            self.trace = parent.child()
        else:
            self.trace = TraceContext.new()

        self._token = _current_trace.set(self.trace)

        self.logger.debug(
            f"span:start:{self.name}",
            tags={"span_name": self.name},
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000

        tags = {"span_name": self.name}
        if exc_type:
            tags["error"] = str(exc_val)

        self.logger.debug(
            f"span:end:{self.name}",
            tags=tags,
            metrics={"duration_ms": duration_ms},
        )

        # Restore previous context
        if self._token:
            _current_trace.reset(self._token)

        return False  # Don't suppress exceptions


class MetricsCollector:
    """
    Collects and aggregates metrics over time.

    Example:
        metrics = MetricsCollector()
        metrics.record("requests", 1, tags={"endpoint": "/api"})
        metrics.record("latency_ms", 42.5)

        summary = metrics.summary()
    """

    def __init__(self):
        self._metrics: Dict[str, List[tuple]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, Any]] = None,
    ):
        """Record a metric value."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []

            self._metrics[name].append((
                time.time(),
                value,
                tags or {},
            ))

    def summary(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for metrics."""
        with self._lock:
            if name:
                values = [v for _, v, _ in self._metrics.get(name, [])]
                if not values:
                    return {}
                return {
                    "count": len(values),
                    "sum": sum(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                }

            # All metrics
            result = {}
            for metric_name, entries in self._metrics.items():
                values = [v for _, v, _ in entries]
                if values:
                    result[metric_name] = {
                        "count": len(values),
                        "sum": sum(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                    }
            return result

    def clear(self, name: Optional[str] = None):
        """Clear recorded metrics."""
        with self._lock:
            if name:
                self._metrics.pop(name, None)
            else:
                self._metrics.clear()


# Global instances
_global_logger: Optional[StructuredLogger] = None
_global_metrics: Optional[MetricsCollector] = None


def get_logger(name: str = "spio") -> StructuredLogger:
    """Get or create a logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name)
    return _global_logger.child(name) if name != "spio" else _global_logger


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics
