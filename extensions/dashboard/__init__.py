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
8. MetricsHistory     - Historical metrics storage
9. (TelemetryBroadcast) - Integrated in DashboardBackend
10. (HeartbeatMonitor) - Via SubstrateState

Terminal (5):
11. TerminalLine      - Single terminal entry
12. (TerminalHistory) - Via TerminalState
13. TerminalProcessor - Command processing
14. CommandRegistry   - Available commands
15. OutputFormatter   - Terminal output formatting

Server (5):
16. ConnectionPool    - Client connection management
17. AuthManager       - Token authentication
18. MessageRouter     - Message routing
19. DashboardBackend  - Main backend interface
20. (DashboardAPI)    - Via DashboardBackend

Panels (5):
21. ResourcePanel     - Resource awareness data
22. LogExplorer       - Log viewing and filtering
23. SettingsPanel     - Dashboard settings
24. Command           - Command model
25. DASHBOARD_VERSION - Version constant

Reference: ASIOS Dashboard React Frontend
"""

from .dashboard_core import (
    # ========================================================================
    # CONSTANTS
    # ========================================================================
    DASHBOARD_VERSION,
    DEFAULT_PORT,
    MAX_TERMINAL_LINES,
    HEARTBEAT_INTERVAL,
    METRICS_INTERVAL,

    # ========================================================================
    # ENUMS
    # ========================================================================
    Theme,
    Panel,
    LineType,
    MessageType,

    # ========================================================================
    # STATE MODELS
    # ========================================================================
    UIState,
    TerminalLine,
    TerminalState,
    TelemetryMetrics,
    SubstrateState,
    DashboardState,

    # ========================================================================
    # STATE MANAGER
    # ========================================================================
    StateManager,

    # ========================================================================
    # TELEMETRY
    # ========================================================================
    TelemetryCollector,
    MetricsHistory,

    # ========================================================================
    # TERMINAL
    # ========================================================================
    Command,
    CommandRegistry,
    TerminalProcessor,
    OutputFormatter,

    # ========================================================================
    # SERVER COMPONENTS
    # ========================================================================
    ConnectionPool,
    AuthManager,
    MessageRouter,

    # ========================================================================
    # PANELS
    # ========================================================================
    ResourcePanel,
    LogExplorer,
    SettingsPanel,

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================
    DashboardBackend,

    # ========================================================================
    # FACTORY FUNCTIONS
    # ========================================================================
    create_dashboard_backend,
    create_state_manager,
    create_telemetry_collector,
    create_command_registry,
)

__all__ = [
    # Constants
    "DASHBOARD_VERSION",
    "DEFAULT_PORT",
    "MAX_TERMINAL_LINES",
    "HEARTBEAT_INTERVAL",
    "METRICS_INTERVAL",

    # Enums
    "Theme",
    "Panel",
    "LineType",
    "MessageType",

    # State Models
    "UIState",
    "TerminalLine",
    "TerminalState",
    "TelemetryMetrics",
    "SubstrateState",
    "DashboardState",

    # State Manager
    "StateManager",

    # Telemetry
    "TelemetryCollector",
    "MetricsHistory",

    # Terminal
    "Command",
    "CommandRegistry",
    "TerminalProcessor",
    "OutputFormatter",

    # Server Components
    "ConnectionPool",
    "AuthManager",
    "MessageRouter",

    # Panels
    "ResourcePanel",
    "LogExplorer",
    "SettingsPanel",

    # Main Interface
    "DashboardBackend",

    # Factory Functions
    "create_dashboard_backend",
    "create_state_manager",
    "create_telemetry_collector",
    "create_command_registry",
]
