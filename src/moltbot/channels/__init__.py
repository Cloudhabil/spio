"""
Moltbot Channel Implementations

Real channel implementations for multi-platform messaging.
"""

from .base import Channel, ChannelType, Message, ChannelConfig
from .terminal import TerminalChannel
from .webhook import WebhookChannel
from .telegram import TelegramChannel
from .discord import DiscordChannel

__all__ = [
    "Channel",
    "ChannelType",
    "Message",
    "ChannelConfig",
    "TerminalChannel",
    "WebhookChannel",
    "TelegramChannel",
    "DiscordChannel",
]
