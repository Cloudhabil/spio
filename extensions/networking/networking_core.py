"""
Networking Extension - WebSocket and Connection Management

Ported from: CLI-main/src/core/ws_connection_manager.py

Implements:
- ConnectionManager: Multi-session WebSocket registry
- Channel-based message broadcasting
- Connection lifecycle management
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class MessageType(Enum):
    """Types of messages that can be sent through connections."""
    TEXT = auto()
    BINARY = auto()
    JSON = auto()
    PING = auto()
    PONG = auto()
    CLOSE = auto()


class ConnectionState(Enum):
    """Connection lifecycle states."""
    CONNECTING = auto()
    CONNECTED = auto()
    CLOSING = auto()
    CLOSED = auto()
    ERROR = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Message:
    """
    A message to be sent through a connection.

    Attributes:
        type: The type of message
        content: The message content
        channel: Optional channel to broadcast to
        session_id: Optional target session
        timestamp: When the message was created
    """
    type: MessageType
    content: Any
    channel: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps({
            "type": self.type.name,
            "content": self.content,
            "channel": self.channel,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str) -> 'Message':
        """Create message from JSON string."""
        parsed = json.loads(data)
        return cls(
            type=MessageType[parsed.get("type", "TEXT")],
            content=parsed.get("content"),
            channel=parsed.get("channel", ""),
            session_id=parsed.get("session_id", ""),
            timestamp=parsed.get("timestamp", time.time()),
        )


@dataclass
class Channel:
    """
    A channel for organizing connections by topic.

    Attributes:
        name: Channel name
        description: Human-readable description
        created_at: When the channel was created
        message_count: Total messages sent through this channel
    """
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    message_count: int = 0

    def increment_count(self):
        """Increment the message count."""
        self.message_count += 1


@dataclass
class Connection:
    """
    Represents a single WebSocket connection.

    Attributes:
        id: Unique connection identifier
        session_id: Session this connection belongs to
        channel: Channel the connection is subscribed to
        state: Current connection state
        websocket: The actual WebSocket object (if any)
        created_at: When the connection was established
        last_activity: When the connection was last active
    """
    id: str
    session_id: str
    channel: str
    state: ConnectionState = ConnectionState.CONNECTING
    websocket: Any = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = time.time()

    @property
    def is_active(self) -> bool:
        """Check if connection is active."""
        return self.state == ConnectionState.CONNECTED


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """
    Multi-session, multi-channel WebSocket connection registry.

    Maintains WebSocket connections grouped by channel and session,
    allowing multiple observers to attach to the same session.

    Usage:
        manager = ConnectionManager()

        # Add a connection
        manager.add("telemetry", "session_123", websocket)

        # Broadcast to all connections in a channel
        await manager.broadcast_text("telemetry", "Hello!")

        # Broadcast to specific session
        await manager.broadcast_text("telemetry", "Hello!", session_id="session_123")

        # Remove a connection
        manager.remove("telemetry", "session_123", websocket)
    """

    def __init__(self):
        # channel -> session_id -> [Connection, ...]
        self._connections: DefaultDict[str, DefaultDict[str, List[Connection]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._channels: Dict[str, Channel] = {}
        self._message_handlers: Dict[str, List[Callable]] = defaultdict(list)

    def add(
        self,
        channel: str,
        session_id: str,
        websocket: Any,
        connection_id: Optional[str] = None
    ) -> Connection:
        """
        Add a WebSocket connection.

        Args:
            channel: Channel to subscribe to
            session_id: Session identifier
            websocket: The WebSocket object
            connection_id: Optional custom connection ID

        Returns:
            The created Connection object
        """
        conn_id = connection_id or f"{channel}_{session_id}_{time.time()}"

        connection = Connection(
            id=conn_id,
            session_id=session_id,
            channel=channel,
            state=ConnectionState.CONNECTED,
            websocket=websocket,
        )

        self._connections[channel][session_id].append(connection)

        # Ensure channel exists
        if channel not in self._channels:
            self._channels[channel] = Channel(name=channel)

        logger.info(f"Added connection {conn_id} to {channel}/{session_id}")
        return connection

    def remove(
        self,
        channel: str,
        session_id: str,
        websocket: Any
    ) -> bool:
        """
        Remove a WebSocket connection.

        Args:
            channel: Channel the connection is in
            session_id: Session identifier
            websocket: The WebSocket object to remove

        Returns:
            True if connection was found and removed
        """
        sessions = self._connections.get(channel)
        if not sessions:
            return False

        connections = sessions.get(session_id, [])

        # Find and remove the connection
        for conn in connections:
            if conn.websocket == websocket:
                connections.remove(conn)
                conn.state = ConnectionState.CLOSED
                logger.info(f"Removed connection {conn.id} from {channel}/{session_id}")

                # Clean up empty structures
                if not connections:
                    sessions.pop(session_id, None)
                if sessions is not None and not sessions:
                    self._connections.pop(channel, None)

                return True

        return False

    def remove_connection(self, connection: Connection) -> bool:
        """Remove a connection by its Connection object."""
        return self.remove(connection.channel, connection.session_id, connection.websocket)

    async def broadcast_text(
        self,
        channel: str,
        message: str,
        session_id: Optional[str] = None
    ) -> int:
        """
        Send a text message to connections.

        Args:
            channel: Channel to broadcast to
            message: The message text
            session_id: Optional specific session to target

        Returns:
            Number of connections that received the message
        """
        if session_id:
            targets = list(self._connections.get(channel, {}).get(session_id, []))
        else:
            targets = [
                conn
                for session_map in self._connections.get(channel, {}).values()
                for conn in session_map
            ]

        sent_count = 0
        disconnected: List[Connection] = []

        for conn in targets:
            if not conn.is_active:
                continue

            try:
                if hasattr(conn.websocket, 'send_text'):
                    await conn.websocket.send_text(message)
                elif hasattr(conn.websocket, 'send'):
                    await conn.websocket.send(message)
                else:
                    # Fallback for mock/test websockets
                    pass
                conn.update_activity()
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to {conn.id}: {e}")
                disconnected.append(conn)

        # Clean up disconnected connections
        for conn in disconnected:
            self.remove_connection(conn)

        # Update channel stats
        if channel in self._channels:
            self._channels[channel].increment_count()

        return sent_count

    async def broadcast_json(
        self,
        channel: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> int:
        """
        Send a JSON message to connections.

        Args:
            channel: Channel to broadcast to
            data: The data to send as JSON
            session_id: Optional specific session to target

        Returns:
            Number of connections that received the message
        """
        message = json.dumps(data)
        return await self.broadcast_text(channel, message, session_id)

    async def broadcast_message(
        self,
        message: Message
    ) -> int:
        """
        Broadcast a Message object.

        Args:
            message: The Message to broadcast

        Returns:
            Number of connections that received the message
        """
        if message.type == MessageType.JSON:
            return await self.broadcast_json(
                message.channel,
                message.content,
                message.session_id or None
            )
        else:
            content = message.content if isinstance(message.content, str) else str(message.content)
            return await self.broadcast_text(
                message.channel,
                content,
                message.session_id or None
            )

    def count(self, channel: Optional[str] = None) -> int:
        """
        Return connection count.

        Args:
            channel: Specific channel to count, or all if None

        Returns:
            Number of connections
        """
        if channel:
            return sum(
                len(conns)
                for conns in self._connections.get(channel, {}).values()
            )
        return sum(
            len(conns)
            for channel_map in self._connections.values()
            for conns in channel_map.values()
        )

    def sessions(self, channel: str) -> Dict[str, List[Connection]]:
        """
        Get session mapping for a channel.

        Args:
            channel: Channel to get sessions for

        Returns:
            Dictionary of session_id -> list of connections
        """
        return dict(self._connections.get(channel, {}))

    def get_channels(self) -> List[Channel]:
        """Get all registered channels."""
        return list(self._channels.values())

    def get_channel(self, name: str) -> Optional[Channel]:
        """Get a specific channel by name."""
        return self._channels.get(name)

    def create_channel(self, name: str, description: str = "") -> Channel:
        """
        Create a new channel.

        Args:
            name: Channel name
            description: Optional description

        Returns:
            The created Channel
        """
        if name not in self._channels:
            self._channels[name] = Channel(name=name, description=description)
        return self._channels[name]

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections."""
        total = 0
        by_channel = {}
        by_state = defaultdict(int)

        for channel, sessions in self._connections.items():
            channel_count = 0
            for session_id, conns in sessions.items():
                for conn in conns:
                    total += 1
                    channel_count += 1
                    by_state[conn.state.name] += 1
            by_channel[channel] = channel_count

        return {
            "total_connections": total,
            "by_channel": by_channel,
            "by_state": dict(by_state),
            "channels": len(self._channels),
        }

    def register_handler(
        self,
        channel: str,
        handler: Callable[[Message], None]
    ) -> None:
        """
        Register a message handler for a channel.

        Args:
            channel: Channel to handle messages for
            handler: Callback function for incoming messages
        """
        self._message_handlers[channel].append(handler)

    async def handle_incoming(
        self,
        channel: str,
        message: str,
        connection: Connection
    ) -> None:
        """
        Handle an incoming message.

        Args:
            channel: Channel the message came from
            message: The raw message text
            connection: The connection that sent it
        """
        connection.update_activity()

        try:
            msg = Message.from_json(message)
        except json.JSONDecodeError:
            msg = Message(
                type=MessageType.TEXT,
                content=message,
                channel=channel,
                session_id=connection.session_id,
            )

        # Call registered handlers
        for handler in self._message_handlers.get(channel, []):
            try:
                handler(msg)
            except Exception as e:
                logger.error(f"Handler error for {channel}: {e}")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_connection_manager: Optional[ConnectionManager] = None


def create_connection_manager() -> ConnectionManager:
    """Create a new ConnectionManager instance."""
    return ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get global ConnectionManager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


__all__ = [
    # Core
    "ConnectionManager", "Connection", "Channel",
    # Message types
    "Message", "MessageType",
    # Factory
    "create_connection_manager", "get_connection_manager",
]
