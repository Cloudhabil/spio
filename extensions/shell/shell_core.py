"""
Shell Core - Terminal Executor and REPL

Based on: CLI-main/src/terminal_executor.py
"""

import logging
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class CommandStatus(Enum):
    """Command execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()


class JobState(Enum):
    """Background job states."""
    RUNNING = auto()
    STOPPED = auto()
    DONE = auto()
    TERMINATED = auto()


class RedirectType(Enum):
    """I/O redirection types."""
    NONE = auto()
    STDOUT = auto()       # >
    STDOUT_APPEND = auto()  # >>
    STDERR = auto()       # 2>
    BOTH = auto()         # &>
    INPUT = auto()        # <
    PIPE = auto()         # |


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Command:
    """Parsed command."""
    raw: str
    executable: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    redirect_stdout: Optional[Path] = None
    redirect_stderr: Optional[Path] = None
    redirect_stdin: Optional[Path] = None
    background: bool = False
    pipe_to: Optional["Command"] = None


@dataclass
class CommandResult:
    """Command execution result."""
    command: str
    status: CommandStatus
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: int = 0
    pid: Optional[int] = None


@dataclass
class Job:
    """Background job."""
    id: int
    command: str
    state: JobState
    pid: int
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None


@dataclass
class Pipeline:
    """Command pipeline."""
    commands: List[Command]
    status: CommandStatus = CommandStatus.PENDING
    results: List[CommandResult] = field(default_factory=list)


# =============================================================================
# COMMAND PARSER
# =============================================================================

class CommandParser:
    """Parses command strings into Command objects."""

    def parse(self, command_str: str) -> Command:
        """Parse a command string."""
        command_str = command_str.strip()

        # Check for background execution
        background = command_str.endswith("&")
        if background:
            command_str = command_str[:-1].strip()

        # Parse redirections
        redirect_stdout = None
        redirect_stderr = None
        redirect_stdin = None

        # Handle >> (append)
        if ">>" in command_str:
            parts = command_str.split(">>", 1)
            command_str = parts[0].strip()
            redirect_stdout = Path(parts[1].strip())

        # Handle > (overwrite)
        elif ">" in command_str and "2>" not in command_str:
            parts = command_str.split(">", 1)
            command_str = parts[0].strip()
            redirect_stdout = Path(parts[1].strip())

        # Handle < (input)
        if "<" in command_str:
            parts = command_str.split("<", 1)
            command_str = parts[0].strip()
            redirect_stdin = Path(parts[1].strip())

        # Parse command and args
        try:
            parts = shlex.split(command_str)
        except ValueError:
            parts = command_str.split()

        executable = parts[0] if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        return Command(
            raw=command_str,
            executable=executable,
            args=args,
            redirect_stdout=redirect_stdout,
            redirect_stderr=redirect_stderr,
            redirect_stdin=redirect_stdin,
            background=background,
        )

    def parse_pipeline(self, command_str: str) -> Pipeline:
        """Parse a pipeline (commands separated by |)."""
        parts = command_str.split("|")
        commands = [self.parse(part.strip()) for part in parts]

        # Link commands
        for i in range(len(commands) - 1):
            commands[i].pipe_to = commands[i + 1]

        return Pipeline(commands=commands)


# =============================================================================
# COMMAND VALIDATOR
# =============================================================================

class CommandValidator:
    """Validates commands for safety."""

    # Allowed commands (whitelist)
    ALLOWED = {
        # Execution
        "python", "python3", "node", "npm", "npx",
        # File operations
        "ls", "dir", "pwd", "cd", "cat", "head", "tail",
        "find", "grep", "awk", "sed",
        # Environment
        "touch", "mkdir", "mv", "cp", "rm", "chmod",
        # Process
        "ps", "top", "kill",
        # Network
        "curl", "wget", "ping",
        # Git
        "git",
        # Misc
        "echo", "printf", "date", "which", "env",
    }

    # Blocked commands (blacklist)
    BLOCKED = {
        "rm -rf /", "rm -rf /*",
        "chmod 777 /",
        "shutdown", "reboot", "halt", "poweroff",
        "mkfs", "fdisk", "dd",
        ":(){ :|:& };:",  # Fork bomb
    }

    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        r"\.\./\.\.",           # Path traversal
        r"\$\([^)]+\)",         # Command substitution
        r"`[^`]+`",             # Backtick execution
        r";\s*rm\s",            # rm after semicolon
        r"\|\s*sh\b",           # Pipe to shell
        r"\|\s*bash\b",         # Pipe to bash
    ]

    def validate(self, command: Command) -> Tuple[bool, str]:
        """Validate a command. Returns (is_valid, reason)."""
        # Check blocked commands
        for blocked in self.BLOCKED:
            if blocked in command.raw:
                return False, f"Blocked command: {blocked}"

        # Check suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, command.raw):
                return False, f"Suspicious pattern detected"

        # Check allowed (if strict mode)
        # For now, allow all that aren't blocked

        return True, ""


# =============================================================================
# TERMINAL EXECUTOR
# =============================================================================

class TerminalExecutor:
    """
    Executes commands with Unix philosophy support.

    Features:
    - Pipes and redirection
    - Background execution
    - Command validation
    - Timeout support
    """

    def __init__(self, working_dir: Path = None):
        self.working_dir = working_dir or Path.cwd()
        self.parser = CommandParser()
        self.validator = CommandValidator()
        self._env = os.environ.copy()
        self._lock = threading.Lock()

    def execute(self, command_str: str, timeout_s: int = 60) -> CommandResult:
        """Execute a command string."""
        command = self.parser.parse(command_str)

        # Validate
        is_valid, reason = self.validator.validate(command)
        if not is_valid:
            return CommandResult(
                command=command_str,
                status=CommandStatus.FAILED,
                exit_code=1,
                stderr=f"Validation failed: {reason}",
            )

        # Execute
        return self._execute_command(command, timeout_s)

    def _execute_command(self, command: Command, timeout_s: int) -> CommandResult:
        """Execute a parsed command."""
        start = time.time()

        try:
            # Build full command
            cmd = [command.executable] + command.args

            # Handle input redirection
            stdin = None
            if command.redirect_stdin and command.redirect_stdin.exists():
                stdin = open(command.redirect_stdin, "r")

            # Execute
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=stdin,
                cwd=str(self.working_dir),
                env=self._env,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout_s)
                exit_code = process.returncode
                status = CommandStatus.COMPLETED if exit_code == 0 else CommandStatus.FAILED
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                exit_code = -1
                status = CommandStatus.TIMEOUT

            # Handle output redirection
            if command.redirect_stdout:
                with open(command.redirect_stdout, "w") as f:
                    f.write(stdout)

            execution_time = int((time.time() - start) * 1000)

            return CommandResult(
                command=command.raw,
                status=status,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time,
                pid=process.pid,
            )

        except FileNotFoundError:
            return CommandResult(
                command=command.raw,
                status=CommandStatus.FAILED,
                exit_code=127,
                stderr=f"Command not found: {command.executable}",
                execution_time_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return CommandResult(
                command=command.raw,
                status=CommandStatus.FAILED,
                exit_code=1,
                stderr=str(e),
                execution_time_ms=int((time.time() - start) * 1000),
            )

    def set_env(self, key: str, value: str):
        """Set environment variable."""
        self._env[key] = value

    def get_env(self, key: str) -> Optional[str]:
        """Get environment variable."""
        return self._env.get(key)

    def change_dir(self, path: Path) -> bool:
        """Change working directory."""
        try:
            new_dir = Path(path).resolve()
            if new_dir.is_dir():
                self.working_dir = new_dir
                return True
        except Exception:
            pass
        return False


class SafeExecutor(TerminalExecutor):
    """Executor with additional safety checks."""

    def __init__(self, working_dir: Path = None, sandbox_paths: List[Path] = None):
        super().__init__(working_dir)
        self.sandbox_paths = sandbox_paths or [self.working_dir]

    def execute(self, command_str: str, timeout_s: int = 60) -> CommandResult:
        """Execute with sandbox path checks."""
        command = self.parser.parse(command_str)

        # Check path safety
        for arg in command.args:
            try:
                path = Path(arg).resolve()
                if path.exists():
                    allowed = any(
                        str(path).startswith(str(sp.resolve()))
                        for sp in self.sandbox_paths
                    )
                    if not allowed:
                        return CommandResult(
                            command=command_str,
                            status=CommandStatus.FAILED,
                            exit_code=1,
                            stderr=f"Path outside sandbox: {path}",
                        )
            except Exception:
                pass

        return super().execute(command_str, timeout_s)


# =============================================================================
# REPL
# =============================================================================

@dataclass
class REPLSession:
    """REPL session state."""
    id: str
    language: str  # "python", "node", etc.
    process: Optional[subprocess.Popen] = None
    history: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


class REPL:
    """Interactive Read-Eval-Print Loop."""

    def __init__(self, executor: TerminalExecutor = None):
        self.executor = executor or TerminalExecutor()
        self._sessions: Dict[str, REPLSession] = {}
        self._history: List[str] = []

    def create_session(self, session_id: str, language: str = "python") -> REPLSession:
        """Create a new REPL session."""
        session = REPLSession(id=session_id, language=language)
        self._sessions[session_id] = session
        return session

    def evaluate(self, session_id: str, code: str) -> str:
        """Evaluate code in a session."""
        session = self._sessions.get(session_id)
        if not session:
            return "Session not found"

        session.history.append(code)
        self._history.append(code)

        # Execute based on language
        if session.language == "python":
            result = self.executor.execute(f"python -c \"{code}\"")
        elif session.language == "node":
            result = self.executor.execute(f"node -e \"{code}\"")
        else:
            result = self.executor.execute(code)

        return result.stdout if result.status == CommandStatus.COMPLETED else result.stderr

    def get_history(self, session_id: str = None) -> List[str]:
        """Get command history."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id].history
        return self._history


# =============================================================================
# JOB CONTROL
# =============================================================================

class BackgroundJob:
    """A background job."""

    def __init__(self, job_id: int, command: str):
        self.id = job_id
        self.command = command
        self.state = JobState.RUNNING
        self.process: Optional[subprocess.Popen] = None
        self.started_at = time.time()
        self.ended_at: Optional[float] = None

    def is_running(self) -> bool:
        if self.process:
            return self.process.poll() is None
        return False


class JobController:
    """Manages background jobs."""

    def __init__(self):
        self._jobs: Dict[int, BackgroundJob] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def create_job(self, command: str) -> BackgroundJob:
        """Create a new background job."""
        with self._lock:
            job = BackgroundJob(self._next_id, command)
            self._jobs[self._next_id] = job
            self._next_id += 1
            return job

    def get_job(self, job_id: int) -> Optional[BackgroundJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[BackgroundJob]:
        """List all jobs."""
        return list(self._jobs.values())

    def kill_job(self, job_id: int) -> bool:
        """Kill a job."""
        job = self._jobs.get(job_id)
        if job and job.process:
            job.process.terminate()
            job.state = JobState.TERMINATED
            job.ended_at = time.time()
            return True
        return False


# =============================================================================
# ENVIRONMENT
# =============================================================================

class VariableManager:
    """Manages shell variables."""

    def __init__(self):
        self._variables: Dict[str, str] = {}
        self._history: List[Tuple[str, str, str]] = []  # (key, old, new)

    def set(self, key: str, value: str):
        """Set a variable."""
        old = self._variables.get(key, "")
        self._variables[key] = value
        self._history.append((key, old, value))

    def get(self, key: str, default: str = "") -> str:
        """Get a variable."""
        return self._variables.get(key, default)

    def expand(self, text: str) -> str:
        """Expand variables in text ($VAR or ${VAR})."""
        result = text
        for key, value in self._variables.items():
            result = result.replace(f"${{{key}}}", value)
            result = result.replace(f"${key}", value)
        return result

    def list_all(self) -> Dict[str, str]:
        """List all variables."""
        return self._variables.copy()


class Environment:
    """Shell environment."""

    def __init__(self):
        self.variables = VariableManager()
        self._cwd = Path.cwd()
        self._home = Path.home()

        # Set default variables
        self.variables.set("HOME", str(self._home))
        self.variables.set("PWD", str(self._cwd))
        self.variables.set("USER", os.getenv("USER", "user"))

    @property
    def cwd(self) -> Path:
        return self._cwd

    def cd(self, path: str) -> bool:
        """Change directory."""
        try:
            new_path = Path(path).expanduser().resolve()
            if new_path.is_dir():
                self._cwd = new_path
                self.variables.set("PWD", str(self._cwd))
                return True
        except Exception:
            pass
        return False


# =============================================================================
# SHELL (UNIFIED INTERFACE)
# =============================================================================

class Shell:
    """Unified shell interface."""

    def __init__(self, working_dir: Path = None):
        self.executor = SafeExecutor(working_dir)
        self.parser = CommandParser()
        self.validator = CommandValidator()
        self.repl = REPL(self.executor)
        self.jobs = JobController()
        self.env = Environment()

    def run(self, command: str, timeout_s: int = 60) -> CommandResult:
        """Run a command."""
        # Expand variables
        expanded = self.env.variables.expand(command)
        return self.executor.execute(expanded, timeout_s)

    def cd(self, path: str) -> bool:
        """Change directory."""
        if self.env.cd(path):
            self.executor.working_dir = self.env.cwd
            return True
        return False

    def set_var(self, key: str, value: str):
        """Set a variable."""
        self.env.variables.set(key, value)

    def get_var(self, key: str) -> str:
        """Get a variable."""
        return self.env.variables.get(key)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_shell: Optional[Shell] = None


def create_shell(working_dir: Path = None) -> Shell:
    return Shell(working_dir)


def create_executor(working_dir: Path = None) -> TerminalExecutor:
    return TerminalExecutor(working_dir)


def get_shell() -> Shell:
    global _shell
    if _shell is None:
        _shell = Shell()
    return _shell
