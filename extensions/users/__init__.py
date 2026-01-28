"""
Users Extension - Unix Permissions and Authorization

Components:
1. AgentUser - User identity with resource limits
2. AgentGroup - Group membership
3. AgentRole - Role-based access control
4. PermissionManager - Permission enforcement
5. Sandbox - Execution isolation

Reference: CLI-main/src/core/unix_permissions.py
"""

from .users_core import (
    # Enums
    AgentRole,
    PermissionBit,
    SandboxLevel,

    # Data classes
    AgentUser,
    AgentGroup,
    ResourceLimits,
    Permission,

    # Permission system
    PermissionManager,
    PermissionChecker,
    ACL,

    # Sandbox
    Sandbox,
    SandboxConfig,

    # Process monitoring
    ProcessMonitor,
    ResourceEnforcer,

    # Main interface
    Users,

    # Factories
    create_users,
    create_user,
    create_group,
    get_users,
)

__all__ = [
    "AgentRole", "PermissionBit", "SandboxLevel",
    "AgentUser", "AgentGroup", "ResourceLimits", "Permission",
    "PermissionManager", "PermissionChecker", "ACL",
    "Sandbox", "SandboxConfig",
    "ProcessMonitor", "ResourceEnforcer",
    "Users",
    "create_users", "create_user", "create_group", "get_users",
]
