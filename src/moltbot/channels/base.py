"""
Moltbot Base Channel Interface

Defines the channel protocol and base implementations.
"""

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Awaitable
from enum import Enum


class ChannelType(Enum):
    """Supported channel types."""
    TERMINAL = "terminal"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    SIGNAL = "signal"
    MATRIX = "matrix"
    IMESSAGE = "imessage"
    LINE = "line"
    TEAMS = "teams"
    WEB = "web"


@dataclass
class Message:
    """A message in the system."""
    id: str
    channel: ChannelType
    sender: str
    content: str
    timestamp: float = field(default_factory=time.time)
    recipient: Optional[str] = None
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChannelConfig:
    """Base configuration for channels."""
    channel_type: ChannelType
    name: str = ""
    enabled: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)


# Message handler type
MessageHandler = Callable[[Message], Awaitable[Optional[str]]]


class Channel(ABC):
    """
    Abstract base class for channel implementations.

    All channels must implement:
    - connect(): Establish connection
    - disconnect(): Close connection
    - send(): Send a message
    - listen(): Start listening for messages
    """

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.connected = False
        self._handlers: List[MessageHandler] = []

    @property
    def channel_type(self) -> ChannelType:
        return self.config.channel_type

    @property
    def name(self) -> str:
        return self.config.name or self.config.channel_type.value

    def on_message(self, handler: MessageHandler) -> "Channel":
        """Register a message handler."""
        self._handlers.append(handler)
        return self

    async def _dispatch(self, message: Message) -> Optional[str]:
        """Dispatch message to all handlers."""
        for handler in self._handlers:
            try:
                response = await handler(message)
                if response is not None:
                    return response
            except Exception as e:
                print(f"Handler error: {e}")
        return None

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the channel. Returns True on success."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the channel. Returns True on success."""
        pass

    @abstractmethod
    async def send(self, recipient: str, content: str, **kwargs) -> bool:
        """Send a message. Returns True on success."""
        pass

    @abstractmethod
    async def listen(self) -> None:
        """Start listening for incoming messages."""
        pass

    async def reply(self, message: Message, content: str) -> bool:
        """Reply to a message."""
        return await self.send(
            recipient=message.sender,
            content=content,
            reply_to=message.id,
        )

    def status(self) -> Dict[str, Any]:
        """Get channel status."""
        return {
            "type": self.channel_type.value,
            "name": self.name,
            "connected": self.connected,
            "enabled": self.config.enabled,
            "handlers": len(self._handlers),
        }
