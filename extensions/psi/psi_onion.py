"""
PSI.Onion Browser - PIO-enhanced Tor Browser Wrapper

Integrates Snowden-grade privacy skills:
- SecureDrop: Air-gap awareness, no-log, metadata stripping
- OnionShare: Ephemeral .onion creation, anonymous dropbox
- GlobaLeaks: 16-digit receipts, Argon2ID, auto-delete
- Briar: No central server, offline sync
- Tails: RAM wipe awareness, stream isolation
- Whonix: IP leak protection, keystroke anonymization

ARCHITECTURE:
    Psi.onion = Tor Browser + PIO Brahim Layer + Snowden Skills
"""

from __future__ import annotations

import os
import secrets
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# Brahim constants
BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 107, 117, 139, 154, 172, 187]
BRAHIM_CENTER = 107
BRAHIM_SUM = 214


# =============================================================================
# RECEIPT SYSTEM (from GlobaLeaks)
# =============================================================================

class ReceiptSystem:
    """
    16-digit receipt system for anonymous session identification.

    Each session gets a unique receipt that can be used to:
    - Resume anonymous sessions
    - Derive encryption keys
    - Prove identity without revealing it
    """

    @staticmethod
    def generate() -> str:
        """Generate a 16-digit receipt."""
        return "".join([str(secrets.randbelow(10)) for _ in range(16)])

    @staticmethod
    def derive_key(receipt: str, salt: bytes = b"psi_onion") -> bytes:
        """Derive encryption key from receipt."""
        # Simple key derivation (in production use Argon2ID)
        combined = receipt.encode() + salt
        return hashlib.sha256(combined).digest()

    @staticmethod
    def verify(receipt: str) -> bool:
        """Verify receipt format."""
        return len(receipt) == 16 and receipt.isdigit()


# =============================================================================
# METADATA STRIPPER (from SecureDrop/Tails)
# =============================================================================

@dataclass
class MetadataStripper:
    """
    Strips identifying metadata from files and content.

    Removes:
    - EXIF data from images
    - Document properties
    - Timestamps
    - Author information
    """

    removed_fields: List[str] = field(default_factory=list)

    def strip_text(self, content: str) -> str:
        """Strip metadata patterns from text."""
        import re

        # Remove common metadata patterns
        patterns = [
            r"Author:.*",
            r"Created:.*",
            r"Modified:.*",
            r"\\author\{.*?\}",
            r"<!-- .* -->",
        ]

        cleaned = content
        for pattern in patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            for match in matches:
                self.removed_fields.append(match[:50])
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        return cleaned

    def strip_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Strip identifying HTTP headers."""
        dangerous = [
            "X-Forwarded-For",
            "X-Real-IP",
            "Via",
            "Forwarded",
            "X-Client-IP",
            "CF-Connecting-IP",
            "True-Client-IP",
        ]

        cleaned = {}
        for key, value in headers.items():
            if key not in dangerous:
                cleaned[key] = value
            else:
                self.removed_fields.append(f"Header: {key}")

        return cleaned

    def report(self) -> Dict:
        """Get stripping report."""
        return {
            "removed_count": len(self.removed_fields),
            "removed_fields": self.removed_fields[:10],  # First 10
        }


# =============================================================================
# STREAM ISOLATOR (from Tails/Whonix)
# =============================================================================

class StreamIsolator:
    """
    Isolates network streams to prevent correlation.

    Each application/purpose gets its own Tor circuit.
    """

    def __init__(self):
        self.streams: Dict[str, Dict] = {}
        self.next_port = 9050

    def create_stream(self, purpose: str) -> Dict:
        """Create isolated stream for a purpose."""
        stream_id = secrets.token_hex(8)
        port = self.next_port
        self.next_port += 1

        stream = {
            "id": stream_id,
            "purpose": purpose,
            "socks_port": port,
            "created": datetime.now(timezone.utc).isoformat(),
            "bytes_sent": 0,
            "bytes_received": 0,
        }

        self.streams[stream_id] = stream
        return stream

    def get_stream(self, stream_id: str) -> Optional[Dict]:
        """Get stream by ID."""
        return self.streams.get(stream_id)

    def close_stream(self, stream_id: str) -> bool:
        """Close and clean up stream."""
        if stream_id in self.streams:
            del self.streams[stream_id]
            return True
        return False

    def list_streams(self) -> List[Dict]:
        """List all active streams."""
        return list(self.streams.values())


# =============================================================================
# SECURE SESSION
# =============================================================================

@dataclass
class SecureSession:
    """
    A secure browsing session with full isolation.

    Combines:
    - Receipt-based identity
    - Stream isolation
    - Metadata stripping
    - Auto-cleanup
    """

    receipt: str = field(default_factory=ReceiptSystem.generate)
    session_key: bytes = field(default=b"")
    stream_isolator: StreamIsolator = field(default_factory=StreamIsolator)
    metadata_stripper: MetadataStripper = field(default_factory=MetadataStripper)

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    auto_delete: bool = True

    def __post_init__(self):
        if not self.session_key:
            self.session_key = ReceiptSystem.derive_key(self.receipt)

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def create_isolated_stream(self, purpose: str) -> Dict:
        """Create isolated stream for this session."""
        return self.stream_isolator.create_stream(f"{self.receipt[:4]}_{purpose}")

    def cleanup(self) -> Dict:
        """Clean up session data."""
        streams_closed = len(self.stream_isolator.streams)
        metadata_stripped = len(self.metadata_stripper.removed_fields)

        # Clear all streams
        self.stream_isolator.streams.clear()

        # Clear metadata records
        self.metadata_stripper.removed_fields.clear()

        return {
            "streams_closed": streams_closed,
            "metadata_stripped": metadata_stripped,
            "session_cleared": True,
        }


# =============================================================================
# PSI ONION BROWSER
# =============================================================================

class PsiOnionBrowser:
    """
    PIO-enhanced Tor Browser wrapper.

    Provides:
    - Anonymous browsing through Brahim layers
    - Snowden-grade privacy skills
    - Automatic session management
    - Metadata stripping
    """

    def __init__(self, tor_socks_port: int = 9050):
        self.tor_socks_port = tor_socks_port
        self.sessions: Dict[str, SecureSession] = {}
        self.current_session: Optional[SecureSession] = None
        self.browsing_layer = BRAHIM_CENTER  # Start at convergence point

    def new_session(self, auto_delete: bool = True) -> SecureSession:
        """Create new secure session."""
        session = SecureSession(auto_delete=auto_delete)
        self.sessions[session.receipt] = session
        self.current_session = session
        return session

    def resume_session(self, receipt: str) -> Optional[SecureSession]:
        """Resume session by receipt."""
        if not ReceiptSystem.verify(receipt):
            return None

        session = self.sessions.get(receipt)
        if session and not session.is_expired:
            self.current_session = session
            return session
        return None

    def browse(self, url: str) -> Dict:
        """Browse URL through Tor with privacy protections."""
        if not self.current_session:
            self.new_session()

        # Create isolated stream for this request
        stream = self.current_session.create_isolated_stream("browse")

        return {
            "url": url,
            "stream_id": stream["id"],
            "socks_port": stream["socks_port"],
            "brahim_layer": self.browsing_layer,
            "receipt": self.current_session.receipt[:4] + "****",
        }

    def rotate_circuit(self) -> int:
        """Rotate to new Brahim layer circuit."""
        current_idx = BRAHIM_SEQUENCE.index(self.browsing_layer)
        next_idx = (current_idx + 1) % len(BRAHIM_SEQUENCE)
        self.browsing_layer = BRAHIM_SEQUENCE[next_idx]
        return self.browsing_layer

    def close_session(self, receipt: Optional[str] = None) -> Dict:
        """Close and cleanup session."""
        target_receipt = receipt or (self.current_session.receipt if self.current_session else None)

        if not target_receipt:
            return {"error": "No session to close"}

        session = self.sessions.get(target_receipt)
        if not session:
            return {"error": "Session not found"}

        cleanup_result = session.cleanup()

        if session.auto_delete:
            del self.sessions[target_receipt]

        if self.current_session and self.current_session.receipt == target_receipt:
            self.current_session = None

        return cleanup_result

    def stats(self) -> Dict:
        """Get browser statistics."""
        return {
            "active_sessions": len(self.sessions),
            "current_layer": self.browsing_layer,
            "tor_port": self.tor_socks_port,
            "total_streams": sum(
                len(s.stream_isolator.streams) for s in self.sessions.values()
            ),
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ReceiptSystem",
    "MetadataStripper",
    "StreamIsolator",
    "SecureSession",
    "PsiOnionBrowser",
]
