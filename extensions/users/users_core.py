"""
Users Core - Unix Permissions and Authorization

Based on: CLI-main/src/core/unix_permissions.py
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AgentRole(Enum):
    """Agent role levels."""
    ROOT = 0        # Full access
    ADMIN = 1       # Administrative access
    STANDARD = 2    # Normal user
    RESTRICTED = 3  # Limited access
    GUEST = 4       # Minimal access


class PermissionBit(Enum):
    """Unix-style permission bits."""
    READ = 4
    WRITE = 2
    EXECUTE = 1


class SandboxLevel(Enum):
    """Sandbox isolation levels."""
    NONE = 0        # No sandbox
    MINIMAL = 1     # Basic restrictions
    STANDARD = 2    # Normal sandbox
    STRICT = 3      # High isolation
    PARANOID = 4    # Maximum isolation


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ResourceLimits:
    """Resource limits for an agent."""
    max_cpu_percent: float = 100.0
    max_memory_mb: int = 4096
    max_processes: int = 10
    max_files: int = 100
    max_network_connections: int = 10
    max_execution_time_s: int = 3600


@dataclass
class AgentUser:
    """Agent user identity."""
    username: str
    uid: int
    gid: int
    role: AgentRole = AgentRole.STANDARD
    home_dir: Path = field(default_factory=lambda: Path("~"))
    groups: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    allowed_paths: List[Path] = field(default_factory=list)
    can_execute: bool = True
    can_network: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentGroup:
    """Agent group."""
    name: str
    gid: int
    members: Set[str] = field(default_factory=set)
    permissions: int = 0o770  # rwxrwx---
    shared_workspace: Optional[Path] = None


@dataclass
class Permission:
    """Permission entry."""
    path: Path
    owner_uid: int
    owner_gid: int
    mode: int = 0o644  # rw-r--r--

    @property
    def mode_string(self) -> str:
        """Get permission string like 'rw-r--r--'."""
        result = ""
        for shift in [6, 3, 0]:  # owner, group, other
            bits = (self.mode >> shift) & 0o7
            result += "r" if bits & 4 else "-"
            result += "w" if bits & 2 else "-"
            result += "x" if bits & 1 else "-"
        return result


# =============================================================================
# ACL (Access Control List)
# =============================================================================

class ACL:
    """Access Control List for a resource."""

    def __init__(self, resource_path: Path):
        self.resource_path = resource_path
        self._entries: Dict[int, int] = {}  # uid -> permission bits
        self._group_entries: Dict[int, int] = {}  # gid -> permission bits

    def grant_user(self, uid: int, permissions: int):
        """Grant permissions to a user."""
        self._entries[uid] = permissions

    def grant_group(self, gid: int, permissions: int):
        """Grant permissions to a group."""
        self._group_entries[gid] = permissions

    def revoke_user(self, uid: int):
        """Revoke user permissions."""
        self._entries.pop(uid, None)

    def check(self, uid: int, gid: int, required: int) -> bool:
        """Check if user has required permissions."""
        # Check user entry
        if uid in self._entries:
            if (self._entries[uid] & required) == required:
                return True

        # Check group entry
        if gid in self._group_entries:
            if (self._group_entries[gid] & required) == required:
                return True

        return False


# =============================================================================
# PERMISSION SYSTEM
# =============================================================================

class PermissionChecker:
    """Checks permissions for operations."""

    def __init__(self):
        self._permissions: Dict[str, Permission] = {}
        self._acls: Dict[str, ACL] = {}

    def set_permission(self, path: Path, owner_uid: int, owner_gid: int, mode: int):
        """Set permission for a path."""
        perm = Permission(path, owner_uid, owner_gid, mode)
        self._permissions[str(path)] = perm

    def check_read(self, user: AgentUser, path: Path) -> bool:
        """Check read permission."""
        return self._check(user, path, PermissionBit.READ.value)

    def check_write(self, user: AgentUser, path: Path) -> bool:
        """Check write permission."""
        return self._check(user, path, PermissionBit.WRITE.value)

    def check_execute(self, user: AgentUser, path: Path) -> bool:
        """Check execute permission."""
        return self._check(user, path, PermissionBit.EXECUTE.value)

    def _check(self, user: AgentUser, path: Path, required_bit: int) -> bool:
        """Internal permission check."""
        # Root can do anything
        if user.role == AgentRole.ROOT:
            return True

        path_str = str(path)
        perm = self._permissions.get(path_str)

        if perm is None:
            # No explicit permission, check role
            return user.role in [AgentRole.ROOT, AgentRole.ADMIN]

        # Check owner
        if perm.owner_uid == user.uid:
            owner_bits = (perm.mode >> 6) & 0o7
            if owner_bits & required_bit:
                return True

        # Check group
        if perm.owner_gid == user.gid:
            group_bits = (perm.mode >> 3) & 0o7
            if group_bits & required_bit:
                return True

        # Check other
        other_bits = perm.mode & 0o7
        return bool(other_bits & required_bit)


class PermissionManager:
    """High-level permission management."""

    def __init__(self):
        self.checker = PermissionChecker()
        self._users: Dict[str, AgentUser] = {}
        self._groups: Dict[str, AgentGroup] = {}
        self._next_uid = 1000
        self._next_gid = 1000
        self._lock = threading.Lock()

    def create_user(
        self,
        username: str,
        role: AgentRole = AgentRole.STANDARD,
        groups: List[str] = None,
    ) -> AgentUser:
        """Create a new user."""
        with self._lock:
            uid = self._next_uid
            self._next_uid += 1

            # Find or create primary group
            gid = self._next_gid
            if groups and groups[0] in self._groups:
                gid = self._groups[groups[0]].gid
            else:
                self._next_gid += 1

            user = AgentUser(
                username=username,
                uid=uid,
                gid=gid,
                role=role,
                groups=groups or [],
                home_dir=Path(f"~/{username}"),
            )
            self._users[username] = user
            return user

    def create_group(self, name: str, members: List[str] = None) -> AgentGroup:
        """Create a new group."""
        with self._lock:
            gid = self._next_gid
            self._next_gid += 1
            group = AgentGroup(
                name=name,
                gid=gid,
                members=set(members or []),
            )
            self._groups[name] = group
            return group

    def get_user(self, username: str) -> Optional[AgentUser]:
        """Get user by username."""
        return self._users.get(username)

    def get_group(self, name: str) -> Optional[AgentGroup]:
        """Get group by name."""
        return self._groups.get(name)

    def authorize(self, user: AgentUser, action: str, resource: Path) -> bool:
        """Authorize an action on a resource."""
        if action == "read":
            return self.checker.check_read(user, resource)
        elif action == "write":
            return self.checker.check_write(user, resource)
        elif action == "execute":
            return self.checker.check_execute(user, resource)
        return False


# =============================================================================
# SANDBOX
# =============================================================================

@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    level: SandboxLevel = SandboxLevel.STANDARD
    allowed_paths: List[Path] = field(default_factory=list)
    blocked_paths: List[Path] = field(default_factory=list)
    allowed_commands: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=list)
    network_allowed: bool = True
    max_execution_time_s: int = 300


class Sandbox:
    """Execution sandbox for agents."""

    # Default blocked commands
    BLOCKED_COMMANDS = [
        "rm -rf /", "chmod 777 /", "shutdown", "reboot",
        "mkfs", "dd", ":(){ :|:& };:",  # Fork bomb
        "nc", "netcat",  # Network tools
    ]

    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self._violations: List[Dict[str, Any]] = []

    def check_path(self, path: Path) -> bool:
        """Check if path is allowed."""
        path_str = str(path.resolve())

        # Check blocked paths
        for blocked in self.config.blocked_paths:
            if path_str.startswith(str(blocked)):
                self._record_violation("path_blocked", path_str)
                return False

        # Check allowed paths (if any specified)
        if self.config.allowed_paths:
            allowed = False
            for allow in self.config.allowed_paths:
                if path_str.startswith(str(allow)):
                    allowed = True
                    break
            if not allowed:
                self._record_violation("path_not_allowed", path_str)
                return False

        return True

    def check_command(self, command: str) -> bool:
        """Check if command is allowed."""
        cmd_lower = command.lower()

        # Check blocked commands
        for blocked in self.BLOCKED_COMMANDS + self.config.blocked_commands:
            if blocked.lower() in cmd_lower:
                self._record_violation("command_blocked", command)
                return False

        # Check allowed commands (if any specified)
        if self.config.allowed_commands:
            allowed = False
            for allow in self.config.allowed_commands:
                if cmd_lower.startswith(allow.lower()):
                    allowed = True
                    break
            if not allowed:
                self._record_violation("command_not_allowed", command)
                return False

        return True

    def _record_violation(self, violation_type: str, details: str):
        """Record a sandbox violation."""
        self._violations.append({
            "type": violation_type,
            "details": details,
            "timestamp": time.time(),
        })
        logger.warning(f"Sandbox violation: {violation_type} - {details}")

    def get_violations(self) -> List[Dict[str, Any]]:
        """Get recorded violations."""
        return self._violations.copy()


# =============================================================================
# PROCESS MONITORING
# =============================================================================

class ProcessMonitor:
    """Monitors process resource usage."""

    def __init__(self, user: AgentUser):
        self.user = user
        self._processes: Dict[int, Dict[str, Any]] = {}

    def register_process(self, pid: int, name: str):
        """Register a process for monitoring."""
        self._processes[pid] = {
            "name": name,
            "started_at": time.time(),
            "cpu_time": 0.0,
            "memory_mb": 0.0,
        }

    def check_limits(self, pid: int) -> bool:
        """Check if process is within limits."""
        if pid not in self._processes:
            return True

        proc = self._processes[pid]
        limits = self.user.limits

        # Check CPU usage (simplified)
        if proc["cpu_time"] > limits.max_execution_time_s:
            return False

        # Check memory
        if proc["memory_mb"] > limits.max_memory_mb:
            return False

        return True


class ResourceEnforcer:
    """Enforces resource limits."""

    def __init__(self):
        self._monitors: Dict[str, ProcessMonitor] = {}

    def create_monitor(self, user: AgentUser) -> ProcessMonitor:
        """Create a process monitor for a user."""
        monitor = ProcessMonitor(user)
        self._monitors[user.username] = monitor
        return monitor

    def enforce(self, username: str, pid: int) -> bool:
        """Enforce limits for a process."""
        monitor = self._monitors.get(username)
        if monitor:
            return monitor.check_limits(pid)
        return True


# =============================================================================
# USERS (UNIFIED INTERFACE)
# =============================================================================

class Users:
    """Unified users interface."""

    def __init__(self):
        self.permissions = PermissionManager()
        self.enforcer = ResourceEnforcer()
        self._default_sandbox = Sandbox()

        # Create default users
        self._init_default_users()

    def _init_default_users(self):
        """Initialize default users."""
        # Root user
        self.permissions.create_user("root", AgentRole.ROOT)
        # System user
        self.permissions.create_user("system", AgentRole.ADMIN)
        # Default agent
        self.permissions.create_user("agent", AgentRole.STANDARD)
        # Guest
        self.permissions.create_user("guest", AgentRole.GUEST)

    def create_user(self, username: str, role: AgentRole = AgentRole.STANDARD) -> AgentUser:
        """Create a new user."""
        return self.permissions.create_user(username, role)

    def get_user(self, username: str) -> Optional[AgentUser]:
        """Get user by username."""
        return self.permissions.get_user(username)

    def authorize(self, username: str, action: str, resource: Path) -> bool:
        """Authorize user action."""
        user = self.get_user(username)
        if not user:
            return False
        return self.permissions.authorize(user, action, resource)

    def create_sandbox(self, config: SandboxConfig = None) -> Sandbox:
        """Create a new sandbox."""
        return Sandbox(config)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_users: Optional[Users] = None


def create_users() -> Users:
    return Users()


def create_user(username: str, role: AgentRole = AgentRole.STANDARD) -> AgentUser:
    users = get_users()
    return users.create_user(username, role)


def create_group(name: str, members: List[str] = None) -> AgentGroup:
    users = get_users()
    return users.permissions.create_group(name, members)


def get_users() -> Users:
    global _users
    if _users is None:
        _users = Users()
    return _users
