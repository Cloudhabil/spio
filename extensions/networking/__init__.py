"""
Networking Extension - WebSocket and Connection Management

Ports from: CLI-main/src/core/ws_connection_manager.py

Implements:
- ConnectionManager: Multi-session WebSocket registry
- Channel-based message broadcasting
- Connection lifecycle management
"""

from .networking_core import (
    # Core
    ConnectionManager, Connection, Channel,

    # Message types
    Message, MessageType,

    # Factory
    create_connection_manager, get_connection_manager,
)

__all__ = [
    "ConnectionManager", "Connection", "Channel",
    "Message", "MessageType",
    "create_connection_manager", "get_connection_manager",
]
