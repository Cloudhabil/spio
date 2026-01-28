"""
Moltbot - Multi-Channel Messaging Gateway

The communication layer of Sovereign PIO.
Handles multi-platform messaging through various channels.

Supported Channels:
- Terminal (local CLI)
- Webhook (HTTP integrations)
- Telegram
- Discord
- WhatsApp (via webhook)
- Slack (via webhook)
- Signal (via webhook)
- Matrix (via webhook)
- And more...
"""

from .channels.base import Channel, ChannelType, Message, ChannelConfig
from .channels.terminal import TerminalChannel
from .channels.webhook import WebhookChannel, WebhookConfig
from .channels.telegram import TelegramChannel, TelegramConfig
from .channels.discord import DiscordChannel, DiscordConfig
from .gateway import Gateway, GatewayConfig

__all__ = [
    # Base
    "Channel",
    "ChannelType",
    "Message",
    "ChannelConfig",
    # Channels
    "TerminalChannel",
    "WebhookChannel",
    "WebhookConfig",
    "TelegramChannel",
    "TelegramConfig",
    "DiscordChannel",
    "DiscordConfig",
    # Gateway
    "Gateway",
    "GatewayConfig",
]
