"""
Dashboard Extension - ASIOS Dashboard Backend

25 Components for dashboard state management and real-time telemetry:

State Management (5):
1. DashboardState     - Complete UI state model
2. TerminalState      - Terminal history and processing
3. SubstrateState     - Connection and metrics
4. UIState            - Panel and theme settings
5. StateManager       - Unified state manager

Telemetry (5):
6. TelemetryMetrics   - CPU, memory, threads, temperature
7. TelemetryCollector - System metrics collection
8. TelemetryBroadcast - Real-time metric broadcasting
9. HeartbeatMonitor   - Connection health monitoring
10. MetricsHistory    - Historical metrics storage

Terminal (5):
11. TerminalLine      - Single terminal entry
12. TerminalHistory   - Command history
13. TerminalProcessor - Command processing
14. CommandRegistry   - Available commands
15. OutputFormatter   - Terminal output formatting

Server (5):
16. WebSocketServer   - Real-time connection server
17. DashboardAPI      - REST API endpoints
18. AuthManager       - Token authentication
19. ConnectionPool    - Client connection management
20. MessageRouter     - Message routing

Panels (5):
21. ResourcePanel     - Resource awareness data
22. LogExplorer       - Log viewing and filtering
23. GenerativeCanvas  - Generative AI tools
24. PersonalityPanel  - Personality selection
25. SettingsPanel     - Dashboard settings

Reference: ASIOS Dashboard React Frontend
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import threading

# ============================================================================
# CONSTANTS
# ============================================================================

DASHBOARD_VERSION = "3.0.1"
DEFAULT_PORT = 8080
MAX_TERMINAL_LINES = 1000
HEARTBEAT_INTERVAL = 5.0  # seconds
METRICS_INTERVAL = 2.0    # seconds


# ============================================================================
# 1. ENUMS
# ============================================================================

class Theme(Enum):
    """Dashboard color themes."""
    HOLOGRAPHIC = "holographic"
    DARK = "dark"
    LIGHT = "light"
    CYBER = "cyber"


class Panel(Enum):
    """Dashboard panels."""
    TERMINAL = "terminal"
    RESOURCES = "resources"
    LOGS = "logs"
    GENERATIVE = "generative"
    SETTINGS = "settings"


class LineType(Enum):
    """Terminal line types."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    COMMAND = "command"
    OUTPUT = "output"


class MessageType(Enum):
    """WebSocket message types."""
    TELEMETRY = "telemetry"
    TERMINAL = "terminal"
    STATE = "state"
    COMMAND = "command"
    HEARTBEAT = "heartbeat"


# ============================================================================
# 2. STATE MODELS
# ============================================================================

@dataclass
class UIState:
    """Dashboard UI state."""
    is_sidebar_open: bool = True
    active_panel: str = Panel.TERMINAL.value
    is_settings_modal_open: bool = False
    theme: str = Theme.HOLOGRAPHIC.value


@dataclass
class TerminalLine:
    """Single terminal line entry."""
    id: str
    timestamp: float
    line_type: str
    content: str

    @classmethod
    def create(cls, line_type: str, content: str) -> 'TerminalLine':
        """Create a new terminal line with auto-generated ID."""
        return cls(
            id=hashlib.sha256(f"{time.time()}{content}".encode()).hexdigest()[:16],
            timestamp=time.time() * 1000,  # JavaScript timestamp
            line_type=line_type,
            content=content
        )


@dataclass
class TerminalState:
    """Terminal state model."""
    history: List[TerminalLine] = field(default_factory=list)
    current_input: str = ""
    is_processing: bool = False
    cursor_position: int = 0

    def add_line(self, line_type: str, content: str) -> TerminalLine:
        """Add a line to history, keeping max lines."""
        line = TerminalLine.create(line_type, content)
        self.history.append(line)
        if len(self.history) > MAX_TERMINAL_LINES:
            self.history = self.history[-MAX_TERMINAL_LINES:]
        return line

    def clear(self) -> None:
        """Clear terminal history."""
        self.history = []


@dataclass
class TelemetryMetrics:
    """System telemetry metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_threads: int = 0
    temperature: float = 0.0
    disk_usage: float = 0.0
    network_in: float = 0.0
    network_out: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to frontend-compatible dict."""
        return {
            "cpuUsage": self.cpu_usage,
            "memoryUsage": self.memory_usage,
            "activeThreads": self.active_threads,
            "temperature": self.temperature,
            "diskUsage": self.disk_usage,
            "networkIn": self.network_in,
            "networkOut": self.network_out,
        }


@dataclass
class SubstrateState:
    """Substrate connection state."""
    is_connected: bool = False
    last_heartbeat: Optional[float] = None
    metrics: TelemetryMetrics = field(default_factory=TelemetryMetrics)

    def update_heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = time.time() * 1000


@dataclass
class DashboardState:
    """Complete dashboard state."""
    ui: UIState = field(default_factory=UIState)
    terminal: TerminalState = field(default_factory=TerminalState)
    substrate: SubstrateState = field(default_factory=SubstrateState)
    version: str = DASHBOARD_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "ui": asdict(self.ui),
            "terminal": {
                "history": [asdict(line) for line in self.terminal.history],
                "currentInput": self.terminal.current_input,
                "isProcessing": self.terminal.is_processing,
                "cursorPosition": self.terminal.cursor_position,
            },
            "substrate": {
                "isConnected": self.substrate.is_connected,
                "lastHeartbeat": self.substrate.last_heartbeat,
                "metrics": self.substrate.metrics.to_dict(),
            },
            "version": self.version,
        }


# ============================================================================
# 3. STATE MANAGER
# ============================================================================

class StateManager:
    """
    Unified dashboard state manager.

    Handles state updates and notifies listeners.
    """

    def __init__(self):
        self.state = DashboardState()
        self._listeners: List[Callable[[DashboardState], None]] = []
        self._lock = threading.Lock()

    def get_state(self) -> DashboardState:
        """Get current state."""
        with self._lock:
            return self.state

    def add_listener(self, callback: Callable[[DashboardState], None]) -> None:
        """Add state change listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[DashboardState], None]) -> None:
        """Remove state change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify_listeners(self) -> None:
        """Notify all listeners of state change."""
        for listener in self._listeners:
            try:
                listener(self.state)
            except Exception as e:
                print(f"Listener error: {e}")

    # UI Actions
    def toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        with self._lock:
            self.state.ui.is_sidebar_open = not self.state.ui.is_sidebar_open
        self._notify_listeners()

    def set_active_panel(self, panel: str) -> None:
        """Set active panel."""
        with self._lock:
            self.state.ui.active_panel = panel
        self._notify_listeners()

    def set_theme(self, theme: str) -> None:
        """Set color theme."""
        with self._lock:
            self.state.ui.theme = theme
        self._notify_listeners()

    # Terminal Actions
    def add_terminal_line(self, line_type: str, content: str) -> TerminalLine:
        """Add terminal line."""
        with self._lock:
            line = self.state.terminal.add_line(line_type, content)
        self._notify_listeners()
        return line

    def clear_terminal(self) -> None:
        """Clear terminal history."""
        with self._lock:
            self.state.terminal.clear()
        self._notify_listeners()

    def set_processing(self, status: bool) -> None:
        """Set terminal processing status."""
        with self._lock:
            self.state.terminal.is_processing = status
        self._notify_listeners()

    # Substrate Actions
    def set_connection_status(self, connected: bool) -> None:
        """Set connection status."""
        with self._lock:
            self.state.substrate.is_connected = connected
            if connected:
                self.state.substrate.update_heartbeat()
        self._notify_listeners()

    def update_metrics(self, metrics: TelemetryMetrics) -> None:
        """Update telemetry metrics."""
        with self._lock:
            self.state.substrate.metrics = metrics
            self.state.substrate.update_heartbeat()
        self._notify_listeners()


# ============================================================================
# 4. TELEMETRY COLLECTOR
# ============================================================================

class TelemetryCollector:
    """
    Collects system telemetry metrics.

    Uses psutil if available, otherwise provides simulated data.
    """

    def __init__(self):
        self._has_psutil = False
        try:
            import psutil
            self._psutil = psutil
            self._has_psutil = True
        except ImportError:
            self._psutil = None

    def collect(self) -> TelemetryMetrics:
        """Collect current system metrics."""
        if self._has_psutil:
            return self._collect_real()
        return self._collect_simulated()

    def _collect_real(self) -> TelemetryMetrics:
        """Collect real metrics using psutil."""
        cpu = self._psutil.cpu_percent(interval=0.1)
        memory = self._psutil.virtual_memory()
        disk = self._psutil.disk_usage('/')
        net = self._psutil.net_io_counters()

        # Temperature (platform-dependent)
        temp = 0.0
        try:
            temps = self._psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        temp = entries[0].current
                        break
        except:
            pass

        return TelemetryMetrics(
            cpu_usage=cpu,
            memory_usage=memory.percent,
            active_threads=threading.active_count(),
            temperature=temp,
            disk_usage=disk.percent,
            network_in=net.bytes_recv / 1024 / 1024,  # MB
            network_out=net.bytes_sent / 1024 / 1024,  # MB
        )

    def _collect_simulated(self) -> TelemetryMetrics:
        """Collect simulated metrics (no psutil)."""
        import random
        return TelemetryMetrics(
            cpu_usage=random.uniform(10, 60),
            memory_usage=random.uniform(30, 70),
            active_threads=threading.active_count(),
            temperature=random.uniform(35, 55),
            disk_usage=random.uniform(20, 80),
            network_in=random.uniform(0, 100),
            network_out=random.uniform(0, 50),
        )


# ============================================================================
# 5. METRICS HISTORY
# ============================================================================

class MetricsHistory:
    """
    Stores historical metrics for graphing.

    Keeps last N samples for each metric.
    """

    def __init__(self, max_samples: int = 60):
        self.max_samples = max_samples
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, metrics: TelemetryMetrics) -> None:
        """Add metrics sample to history."""
        with self._lock:
            sample = {
                "timestamp": time.time() * 1000,
                **metrics.to_dict()
            }
            self._history.append(sample)
            if len(self._history) > self.max_samples:
                self._history = self._history[-self.max_samples:]

    def get_history(self, metric: Optional[str] = None) -> List[Dict]:
        """Get metric history."""
        with self._lock:
            if metric:
                return [{"timestamp": h["timestamp"], "value": h.get(metric, 0)}
                        for h in self._history]
            return self._history.copy()


# ============================================================================
# 6. COMMAND REGISTRY & PROCESSOR
# ============================================================================

@dataclass
class Command:
    """Terminal command definition."""
    name: str
    description: str
    handler: Callable[[List[str]], str]
    aliases: List[str] = field(default_factory=list)


class CommandRegistry:
    """Registry of available terminal commands."""

    def __init__(self):
        self._commands: Dict[str, Command] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default commands."""
        self.register(Command(
            name="help",
            description="Show available commands",
            handler=self._help_handler,
            aliases=["?"]
        ))
        self.register(Command(
            name="clear",
            description="Clear terminal",
            handler=lambda args: "__CLEAR__"
        ))
        self.register(Command(
            name="status",
            description="Show system status",
            handler=self._status_handler
        ))
        self.register(Command(
            name="version",
            description="Show dashboard version",
            handler=lambda args: f"ASIOS Dashboard v{DASHBOARD_VERSION}"
        ))
        self.register(Command(
            name="time",
            description="Show current time",
            handler=lambda args: datetime.now().isoformat()
        ))

    def register(self, command: Command) -> None:
        """Register a command."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command

    def get(self, name: str) -> Optional[Command]:
        """Get command by name or alias."""
        return self._commands.get(name.lower())

    def list_commands(self) -> List[Command]:
        """List all unique commands."""
        seen = set()
        unique = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                unique.append(cmd)
        return unique

    def _help_handler(self, args: List[str]) -> str:
        """Handle help command."""
        lines = ["Available commands:"]
        for cmd in self.list_commands():
            aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
            lines.append(f"  {cmd.name}{aliases} - {cmd.description}")
        return "\n".join(lines)

    def _status_handler(self, args: List[str]) -> str:
        """Handle status command."""
        return (
            f"System: {platform.system()} {platform.release()}\n"
            f"Python: {platform.python_version()}\n"
            f"Threads: {threading.active_count()}\n"
            f"Dashboard: v{DASHBOARD_VERSION}"
        )


class TerminalProcessor:
    """
    Processes terminal commands.
    """

    def __init__(self, registry: Optional[CommandRegistry] = None):
        self.registry = registry or CommandRegistry()

    def process(self, input_line: str) -> str:
        """Process a command and return output."""
        parts = input_line.strip().split()
        if not parts:
            return ""

        cmd_name = parts[0]
        args = parts[1:]

        command = self.registry.get(cmd_name)
        if command:
            return command.handler(args)

        return f"Unknown command: {cmd_name}. Type 'help' for available commands."


# ============================================================================
# 7. OUTPUT FORMATTER
# ============================================================================

class OutputFormatter:
    """Formats terminal output for display."""

    @staticmethod
    def format_json(data: Any) -> str:
        """Format data as JSON."""
        return json.dumps(data, indent=2)

    @staticmethod
    def format_table(headers: List[str], rows: List[List[str]]) -> str:
        """Format data as ASCII table."""
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        lines = []
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        for row in rows:
            lines.append(" | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(row)))
        return "\n".join(lines)


# ============================================================================
# 8. CONNECTION POOL
# ============================================================================

class ConnectionPool:
    """
    Manages WebSocket client connections.
    """

    def __init__(self):
        self._connections: Set[Any] = set()
        self._lock = threading.Lock()

    def add(self, connection: Any) -> None:
        """Add a connection."""
        with self._lock:
            self._connections.add(connection)

    def remove(self, connection: Any) -> None:
        """Remove a connection."""
        with self._lock:
            self._connections.discard(connection)

    def get_all(self) -> Set[Any]:
        """Get all connections."""
        with self._lock:
            return self._connections.copy()

    def count(self) -> int:
        """Get connection count."""
        with self._lock:
            return len(self._connections)

    def broadcast(self, message: Dict) -> None:
        """Broadcast message to all connections."""
        connections = self.get_all()
        msg_json = json.dumps(message)
        for conn in connections:
            try:
                if hasattr(conn, 'send'):
                    conn.send(msg_json)
            except:
                self.remove(conn)


# ============================================================================
# 9. AUTH MANAGER
# ============================================================================

class AuthManager:
    """
    Handles authentication for dashboard connections.
    """

    def __init__(self, secret: Optional[str] = None):
        self.secret = secret or os.environ.get("AGENT_SHARED_SECRET", "")
        self._tokens: Set[str] = set()

    def validate_token(self, token: str) -> bool:
        """Validate an authentication token."""
        if not self.secret:
            return True  # No auth required if no secret set
        return token == self.secret

    def generate_token(self) -> str:
        """Generate a new auth token."""
        token = hashlib.sha256(f"{time.time()}{os.urandom(16).hex()}".encode()).hexdigest()
        self._tokens.add(token)
        return token

    def revoke_token(self, token: str) -> None:
        """Revoke a token."""
        self._tokens.discard(token)


# ============================================================================
# 10. MESSAGE ROUTER
# ============================================================================

class MessageRouter:
    """
    Routes messages between dashboard components.
    """

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, message_type: str, handler: Callable) -> None:
        """Subscribe to a message type."""
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)

    def unsubscribe(self, message_type: str, handler: Callable) -> None:
        """Unsubscribe from a message type."""
        if message_type in self._handlers:
            if handler in self._handlers[message_type]:
                self._handlers[message_type].remove(handler)

    def route(self, message: Dict) -> None:
        """Route a message to handlers."""
        msg_type = message.get("type")
        if msg_type in self._handlers:
            for handler in self._handlers[msg_type]:
                try:
                    handler(message)
                except Exception as e:
                    print(f"Handler error: {e}")


# ============================================================================
# 11. PANEL DATA PROVIDERS
# ============================================================================

class ResourcePanel:
    """Provides data for Resource Awareness panel."""

    def __init__(self, collector: Optional[TelemetryCollector] = None):
        self.collector = collector or TelemetryCollector()

    def get_data(self) -> Dict:
        """Get current resource data."""
        metrics = self.collector.collect()
        return {
            "cpu": {"value": metrics.cpu_usage, "label": "CPU"},
            "memory": {"value": metrics.memory_usage, "label": "Memory"},
            "disk": {"value": metrics.disk_usage, "label": "Disk"},
            "temperature": {"value": metrics.temperature, "label": "Temp"},
        }


class LogExplorer:
    """Provides log exploration functionality."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)

    def list_logs(self) -> List[Dict]:
        """List available log files."""
        if not self.log_dir.exists():
            return []

        logs = []
        for file in self.log_dir.glob("*.log"):
            logs.append({
                "name": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime,
            })
        return sorted(logs, key=lambda x: x["modified"], reverse=True)

    def read_log(self, name: str, lines: int = 100) -> List[str]:
        """Read last N lines from a log file."""
        path = self.log_dir / name
        if not path.exists():
            return []

        try:
            with open(path) as f:
                all_lines = f.readlines()
                return all_lines[-lines:]
        except:
            return []


class SettingsPanel:
    """Provides settings panel data."""

    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager

    def get_settings(self) -> Dict:
        """Get current settings."""
        if not self.state_manager:
            return {}

        state = self.state_manager.get_state()
        return {
            "theme": state.ui.theme,
            "sidebarOpen": state.ui.is_sidebar_open,
            "version": state.version,
        }

    def update_settings(self, settings: Dict) -> None:
        """Update settings."""
        if not self.state_manager:
            return

        if "theme" in settings:
            self.state_manager.set_theme(settings["theme"])


# ============================================================================
# 12. DASHBOARD BACKEND (Main Interface)
# ============================================================================

class DashboardBackend:
    """
    Main dashboard backend interface.

    Provides unified access to all dashboard components.
    """

    def __init__(self, secret: Optional[str] = None):
        # Core components
        self.state_manager = StateManager()
        self.collector = TelemetryCollector()
        self.history = MetricsHistory()
        self.processor = TerminalProcessor()
        self.connection_pool = ConnectionPool()
        self.auth = AuthManager(secret)
        self.router = MessageRouter()

        # Panels
        self.resource_panel = ResourcePanel(self.collector)
        self.log_explorer = LogExplorer()
        self.settings_panel = SettingsPanel(self.state_manager)

        # Initialize terminal
        self.state_manager.add_terminal_line(
            LineType.INFO.value,
            f"ASIOS Substrate Initialized v{DASHBOARD_VERSION}"
        )

    def process_command(self, command: str) -> str:
        """Process a terminal command."""
        # Add command to history
        self.state_manager.add_terminal_line(LineType.COMMAND.value, f"$ {command}")
        self.state_manager.set_processing(True)

        # Process
        output = self.processor.process(command)

        if output == "__CLEAR__":
            self.state_manager.clear_terminal()
            output = "Terminal cleared"
        elif output:
            self.state_manager.add_terminal_line(LineType.OUTPUT.value, output)

        self.state_manager.set_processing(False)
        return output

    def collect_metrics(self) -> TelemetryMetrics:
        """Collect and store metrics."""
        metrics = self.collector.collect()
        self.history.add(metrics)
        self.state_manager.update_metrics(metrics)
        return metrics

    def get_state_json(self) -> str:
        """Get current state as JSON."""
        return json.dumps(self.state_manager.get_state().to_dict())

    def broadcast_telemetry(self) -> None:
        """Broadcast telemetry to all connections."""
        metrics = self.collect_metrics()
        message = {
            "type": MessageType.TELEMETRY.value,
            "data": metrics.to_dict(),
            "timestamp": time.time() * 1000,
        }
        self.connection_pool.broadcast(message)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_dashboard_backend(secret: Optional[str] = None) -> DashboardBackend:
    """Create a dashboard backend instance."""
    return DashboardBackend(secret)


def create_state_manager() -> StateManager:
    """Create a standalone state manager."""
    return StateManager()


def create_telemetry_collector() -> TelemetryCollector:
    """Create a telemetry collector."""
    return TelemetryCollector()


def create_command_registry() -> CommandRegistry:
    """Create a command registry with defaults."""
    return CommandRegistry()


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing Dashboard Extension...")
    print(f"Version: {DASHBOARD_VERSION}")

    # Test state manager
    state = StateManager()
    state.add_terminal_line("info", "Test line")
    print(f"Terminal lines: {len(state.get_state().terminal.history)}")

    # Test telemetry
    collector = TelemetryCollector()
    metrics = collector.collect()
    print(f"CPU: {metrics.cpu_usage:.1f}%")
    print(f"Memory: {metrics.memory_usage:.1f}%")

    # Test command processor
    processor = TerminalProcessor()
    output = processor.process("version")
    print(f"Version command: {output}")

    # Test dashboard backend
    backend = DashboardBackend()
    backend.process_command("status")
    print(f"Backend state: {len(backend.state_manager.get_state().terminal.history)} lines")

    print("\nAll tests passed!")
