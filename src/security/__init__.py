"""
Sovereign PIO Security Module

Defense-in-depth security controls for the autonomous agent platform.

Security Layers:
1. Command Validation - Blacklist + heuristic analysis
2. Filesystem Sandboxing - Zone-based access control
3. Multi-tier Authentication - Local/network/token
4. Tool Policies - Profile-based authorization
5. Circuit Breaker - Automatic failure isolation
"""

import re
import shlex
from pathlib import Path
from dataclasses import dataclass

__all__ = [
    "SecurityViolation",
    "SandboxViolation",
    "CommandValidator",
    "FilesystemGuard",
]


class SecurityViolation(Exception):
    """Raised when a security policy is violated."""
    pass


class SandboxViolation(SecurityViolation):
    """Raised when a sandbox boundary is violated."""
    pass


# Dangerous binaries that should never be executed
BLOCKED_BINARIES = frozenset({
    "shutdown", "reboot", "poweroff", "halt",
    "mkfs", "fdisk", "dd",
    "nc", "netcat", "ncat",
    "sh", "bash", "zsh", "dash", "fish",
    "rm",  # Only when combined with dangerous flags
})

# Suspicious argument patterns
SUSPICIOUS_PATTERNS = [
    r"/etc/shadow",
    r"/etc/passwd",
    r"/etc/sudoers",
    r"/root/",
    r"/dev/sda",
    r"/dev/mem",
    r"\.\./",  # Path traversal
    r"\$\(",   # Command substitution
    r"`",      # Backtick substitution
]


@dataclass
class CommandValidator:
    """
    Validates commands before execution.

    Implements blacklist and heuristic-based detection
    to prevent dangerous operations.
    """

    blocked_binaries: frozenset = BLOCKED_BINARIES
    suspicious_patterns: list = None

    def __post_init__(self):
        if self.suspicious_patterns is None:
            self.suspicious_patterns = SUSPICIOUS_PATTERNS

    def validate(self, command: str) -> bool:
        """
        Validate a command for security.

        Args:
            command: The command string to validate

        Returns:
            True if command is safe

        Raises:
            SecurityViolation: If command violates security policy
        """
        # Parse command
        try:
            parts = shlex.split(command, posix=True)
        except ValueError as e:
            raise SecurityViolation(f"Malformed command: {e}")

        if not parts:
            return True

        # Check binary
        binary = Path(parts[0]).name
        if binary in self.blocked_binaries:
            raise SecurityViolation(f"Blocked binary: {binary}")

        # Check for dangerous rm usage
        if binary == "rm" and any(f in parts for f in ["-rf", "-fr", "--recursive"]):
            raise SecurityViolation("Dangerous rm command blocked")

        # Check suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, command):
                raise SecurityViolation(f"Suspicious pattern detected: {pattern}")

        return True


@dataclass
class FilesystemGuard:
    """
    Filesystem sandbox enforcing zone-based access control.

    Read zones: Areas where reading is permitted
    Write zones: Areas where writing is permitted
    """

    project_root: Path = None

    # Default allowed zones (relative to project root)
    read_zones: tuple = (
        "src/",
        "config/",
        "docs/",
        "logs/",
        "README.md",
    )

    write_zones: tuple = (
        "data/",
        "logs/",
        "temp/",
        "outputs/",
    )

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = Path.cwd()
        self.project_root = Path(self.project_root).resolve()

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve and validate a path."""
        resolved = Path(path).resolve()

        # Ensure path is within project root
        try:
            resolved.relative_to(self.project_root)
        except ValueError:
            raise SandboxViolation(f"Path outside project root: {path}")

        return resolved

    def can_read(self, path: str | Path) -> bool:
        """Check if reading from path is allowed."""
        resolved = self._resolve_path(path)
        relative = resolved.relative_to(self.project_root)

        for zone in self.read_zones:
            if str(relative).startswith(zone) or str(relative) == zone.rstrip("/"):
                return True

        return False

    def can_write(self, path: str | Path) -> bool:
        """Check if writing to path is allowed."""
        resolved = self._resolve_path(path)
        relative = resolved.relative_to(self.project_root)

        for zone in self.write_zones:
            if str(relative).startswith(zone):
                return True

        return False

    def validate_read(self, path: str | Path):
        """Validate read access, raise if not allowed."""
        if not self.can_read(path):
            raise SandboxViolation(f"Read access denied: {path}")

    def validate_write(self, path: str | Path):
        """Validate write access, raise if not allowed."""
        if not self.can_write(path):
            raise SandboxViolation(f"Write access denied: {path}")
