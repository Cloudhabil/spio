"""
Shell Extension - Terminal Executor and REPL

Components:
1. TerminalExecutor - Command execution with Unix philosophy
2. CommandParser - Command parsing and validation
3. REPL - Interactive read-eval-print loop
4. JobControl - Background job management
5. Environment - Environment variables

Reference: CLI-main/src/terminal_executor.py
"""

from .shell_core import (
    # Enums
    CommandStatus,
    JobState,
    RedirectType,

    # Data classes
    Command,
    CommandResult,
    Job,
    Pipeline,

    # Parser
    CommandParser,
    CommandValidator,

    # Executor
    TerminalExecutor,
    SafeExecutor,

    # REPL
    REPL,
    REPLSession,

    # Job control
    JobController,
    BackgroundJob,

    # Environment
    Environment,
    VariableManager,

    # Main interface
    Shell,

    # Factories
    create_shell,
    create_executor,
    get_shell,
)

__all__ = [
    "CommandStatus", "JobState", "RedirectType",
    "Command", "CommandResult", "Job", "Pipeline",
    "CommandParser", "CommandValidator",
    "TerminalExecutor", "SafeExecutor",
    "REPL", "REPLSession",
    "JobController", "BackgroundJob",
    "Environment", "VariableManager",
    "Shell",
    "create_shell", "create_executor", "get_shell",
]
