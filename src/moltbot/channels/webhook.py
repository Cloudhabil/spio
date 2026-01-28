"""
Webhook Channel

HTTP webhook-based channel for integrations with external services.
Supports both incoming webhooks (receive) and outgoing webhooks (send).
"""

import asyncio
import uuid
import json
import hmac
import hashlib
from typing import Optional, Dict, Any
from dataclasses import dataclass

import httpx

from .base import Channel, ChannelConfig, ChannelType, Message


@dataclass
class WebhookConfig(ChannelConfig):
    """Webhook-specific configuration."""
    incoming_port: int = 8080
    incoming_path: str = "/webhook"
    outgoing_url: Optional[str] = None
    secret: Optional[str] = None  # For signature verification

    def __post_init__(self):
        self.channel_type = ChannelType.WEBHOOK


class WebhookChannel(Channel):
    """
    Webhook-based channel for HTTP integrations.

    Features:
    - Incoming webhook server (receives messages)
    - Outgoing webhook client (sends messages)
    - HMAC signature verification
    - Async HTTP handling
    """

    def __init__(self, config: Optional[WebhookConfig] = None):
        if config is None:
            config = WebhookConfig(
                channel_type=ChannelType.WEBHOOK,
                name="webhook",
            )
        super().__init__(config)
        self.webhook_config: WebhookConfig = config
        self._server = None
        self._client = httpx.AsyncClient(timeout=30.0)

    async def connect(self) -> bool:
        """Start webhook server if incoming is configured."""
        try:
            # Import here to avoid dependency if not used
            from aiohttp import web

            app = web.Application()
            app.router.add_post(self.webhook_config.incoming_path, self._handle_incoming)

            runner = web.AppRunner(app)
            await runner.setup()

            self._server = web.TCPSite(runner, "0.0.0.0", self.webhook_config.incoming_port)
            await self._server.start()

            self.connected = True
            print(f"Webhook listening on port {self.webhook_config.incoming_port}")
            return True

        except ImportError:
            # aiohttp not available, still allow outgoing
            print("aiohttp not installed - webhook server disabled")
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to start webhook server: {e}")
            return False

    async def disconnect(self) -> bool:
        """Stop webhook server."""
        if self._server:
            await self._server.stop()
        await self._client.aclose()
        self.connected = False
        return True

    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify HMAC signature."""
        if not self.webhook_config.secret:
            return True  # No secret configured

        expected = hmac.new(
            self.webhook_config.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(f"sha256={expected}", signature)

    async def _handle_incoming(self, request) -> Any:
        """Handle incoming webhook request."""
        from aiohttp import web

        try:
            # Read body
            body = await request.read()

            # Verify signature if configured
            signature = request.headers.get("X-Signature", "")
            if self.webhook_config.secret and not self._verify_signature(body, signature):
                return web.Response(status=401, text="Invalid signature")

            # Parse JSON
            data = json.loads(body)

            # Extract message fields
            message = Message(
                id=data.get("id", str(uuid.uuid4())),
                channel=ChannelType.WEBHOOK,
                sender=data.get("sender", "webhook"),
                content=data.get("content", data.get("text", "")),
                metadata=data.get("metadata", {}),
            )

            # Dispatch to handlers
            response = await self._dispatch(message)

            # Return response
            return web.json_response({
                "status": "ok",
                "response": response,
            })

        except json.JSONDecodeError:
            return web.Response(status=400, text="Invalid JSON")
        except Exception as e:
            return web.Response(status=500, text=str(e))

    async def send(self, recipient: str, content: str, **kwargs) -> bool:
        """Send message via outgoing webhook."""
        if not self.webhook_config.outgoing_url:
            print(f"[Webhook] No outgoing URL configured: {content}")
            return False

        try:
            payload = {
                "recipient": recipient,
                "content": content,
                "metadata": kwargs.get("metadata", {}),
            }

            # Add signature if secret configured
            headers = {"Content-Type": "application/json"}
            if self.webhook_config.secret:
                body = json.dumps(payload).encode()
                sig = hmac.new(
                    self.webhook_config.secret.encode(),
                    body,
                    hashlib.sha256
                ).hexdigest()
                headers["X-Signature"] = f"sha256={sig}"

            response = await self._client.post(
                self.webhook_config.outgoing_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return True

        except Exception as e:
            print(f"Webhook send failed: {e}")
            return False

    async def listen(self) -> None:
        """Keep webhook server running."""
        while self.connected:
            await asyncio.sleep(1)
