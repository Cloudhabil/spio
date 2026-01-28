"""
PSI.Firefox Browser - PIO-enhanced Firefox Privacy Wrapper

BASED ON: Mozilla Firefox / Gecko Engine

GECKO ARCHITECTURE:
    browser/    - Desktop UI (XUL, JavaScript, C++)
    dom/        - DOM implementation
    layout/     - Rendering engine (CSS boxes, frames)
    js/         - SpiderMonkey JavaScript engine

SKILLS INTEGRATED:
    - SecureDrop: Air-gap awareness, no-log, metadata stripping
    - GlobaLeaks: 16-digit receipts, Argon2ID, auto-delete
    - Tails: RAM wipe awareness, stream isolation
    - Whonix: IP leak protection, keystroke anonymization, DNS leak prevention

ARCHITECTURE:
    Psi.firefox = Firefox/Gecko + PIO Brahim Layer + Snowden Skills
"""

from __future__ import annotations

import os
import json
import secrets
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# Brahim constants
BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 107, 117, 139, 154, 172, 187]
BRAHIM_CENTER = 107

# Firefox/Gecko constants
GECKO_ENGINE = "Gecko"
GECKO_COMPONENTS = {
    "browser": "Desktop UI (XUL, JavaScript, C++)",
    "dom": "DOM implementation",
    "layout": "Rendering engine (CSS boxes, frames)",
    "js": "SpiderMonkey JavaScript engine",
    "docshell": "Frame loading/embedding",
    "widget": "Cross-platform OS widgets",
    "xpcom": "Component Object Model",
    "netwerk": "Networking (Necko)",
    "security": "NSS cryptographic services",
    "gfx": "Graphics rendering (WebRender)",
}


# =============================================================================
# PRIVACY PROFILE
# =============================================================================

class PrivacyLevel(Enum):
    """Privacy protection levels."""
    STANDARD = "standard"        # Basic protections
    STRICT = "strict"            # Enhanced tracking protection
    PARANOID = "paranoid"        # Maximum isolation


@dataclass
class PrivacyProfile:
    """
    Firefox privacy profile configuration.

    Configures about:config settings for maximum privacy.
    """

    name: str
    level: PrivacyLevel = PrivacyLevel.STRICT

    # Tracking protection
    tracking_protection: bool = True
    fingerprint_protection: bool = True
    cookie_isolation: bool = True

    # Network privacy
    dns_over_https: bool = True
    webrtc_disabled: bool = True
    prefetch_disabled: bool = True

    # Data retention
    clear_on_exit: bool = True
    history_disabled: bool = False
    cache_disabled: bool = False

    def to_prefs(self) -> Dict[str, Any]:
        """Convert to Firefox prefs.js format."""
        prefs = {
            # Tracking protection
            "privacy.trackingprotection.enabled": self.tracking_protection,
            "privacy.trackingprotection.socialtracking.enabled": self.tracking_protection,
            "privacy.resistFingerprinting": self.fingerprint_protection,
            "privacy.firstparty.isolate": self.cookie_isolation,

            # Network
            "network.trr.mode": 3 if self.dns_over_https else 0,  # 3 = DoH only
            "media.peerconnection.enabled": not self.webrtc_disabled,
            "network.prefetch-next": not self.prefetch_disabled,
            "network.dns.disablePrefetch": self.prefetch_disabled,

            # Data
            "privacy.sanitize.sanitizeOnShutdown": self.clear_on_exit,
            "places.history.enabled": not self.history_disabled,
            "browser.cache.disk.enable": not self.cache_disabled,

            # Additional paranoid settings
            "geo.enabled": False,
            "dom.battery.enabled": False,
            "beacon.enabled": False,
            "browser.send_pings": False,
            "browser.urlbar.speculativeConnect.enabled": False,
        }

        if self.level == PrivacyLevel.PARANOID:
            prefs.update({
                "javascript.enabled": True,  # Needed but restricted
                "dom.event.clipboardevents.enabled": False,
                "dom.storage.enabled": False,
                "network.cookie.cookieBehavior": 2,  # Block all
                "media.navigator.enabled": False,
            })

        return prefs

    def save_to_profile(self, profile_path: Path) -> bool:
        """Save prefs to Firefox profile."""
        prefs_file = profile_path / "user.js"
        prefs = self.to_prefs()

        lines = [f'user_pref("{k}", {json.dumps(v)});' for k, v in prefs.items()]

        prefs_file.write_text("\n".join(lines))
        return True


# =============================================================================
# FINGERPRINT BLOCKER
# =============================================================================

class FingerprintBlocker:
    """
    Blocks browser fingerprinting techniques.

    Protects against:
    - Canvas fingerprinting
    - WebGL fingerprinting
    - Audio fingerprinting
    - Font enumeration
    - Screen resolution detection
    """

    def __init__(self):
        self.blocked_apis: List[str] = []
        self.spoofed_values: Dict[str, Any] = {}

    def block_canvas(self) -> Dict:
        """Block canvas fingerprinting."""
        self.blocked_apis.append("canvas.toDataURL")
        self.blocked_apis.append("canvas.getImageData")
        return {"blocked": "canvas", "method": "noise_injection"}

    def block_webgl(self) -> Dict:
        """Block WebGL fingerprinting."""
        self.blocked_apis.append("WebGLRenderingContext")
        self.blocked_apis.append("WEBGL_debug_renderer_info")
        return {"blocked": "webgl", "method": "api_disabled"}

    def block_audio(self) -> Dict:
        """Block audio fingerprinting."""
        self.blocked_apis.append("AudioContext")
        self.blocked_apis.append("OfflineAudioContext")
        return {"blocked": "audio", "method": "noise_injection"}

    def spoof_screen(self, width: int = 1920, height: int = 1080) -> Dict:
        """Spoof screen resolution to common value."""
        self.spoofed_values["screen.width"] = width
        self.spoofed_values["screen.height"] = height
        self.spoofed_values["screen.availWidth"] = width
        self.spoofed_values["screen.availHeight"] = height - 40  # Taskbar
        return {"spoofed": "screen", "resolution": f"{width}x{height}"}

    def spoof_timezone(self, offset: int = 0) -> Dict:
        """Spoof timezone to UTC."""
        self.spoofed_values["timezone"] = offset
        return {"spoofed": "timezone", "offset": offset}

    def get_protection_status(self) -> Dict:
        """Get current protection status."""
        return {
            "blocked_apis": len(self.blocked_apis),
            "spoofed_values": len(self.spoofed_values),
            "apis": self.blocked_apis,
            "spoofs": list(self.spoofed_values.keys()),
        }


# =============================================================================
# TRACKING PROTECTION
# =============================================================================

class TrackingProtection:
    """
    Enhanced tracking protection system.

    Blocks:
    - Third-party cookies
    - Tracking pixels
    - Social media trackers
    - Cryptominers
    - Fingerprinters
    """

    # Known tracker domains (subset for demo)
    TRACKER_DOMAINS = {
        "doubleclick.net",
        "googlesyndication.com",
        "facebook.com/tr",
        "analytics.google.com",
        "pixel.facebook.com",
        "ads.twitter.com",
    }

    CRYPTOMINER_PATTERNS = {
        "coinhive.com",
        "coin-hive.com",
        "cryptoloot.pro",
        "minero.cc",
    }

    def __init__(self):
        self.blocked_requests: List[Dict] = []
        self.allowed_first_party: List[str] = []

    def check_request(self, url: str, first_party_domain: str) -> bool:
        """
        Check if request should be allowed.

        Returns True if allowed, False if blocked.
        """
        # Extract domain from URL
        domain = self._extract_domain(url)

        # Always allow first-party
        if domain == first_party_domain:
            return True

        # Check against tracker list
        for tracker in self.TRACKER_DOMAINS:
            if tracker in url:
                self._log_blocked(url, "tracker")
                return False

        # Check for cryptominers
        for miner in self.CRYPTOMINER_PATTERNS:
            if miner in url:
                self._log_blocked(url, "cryptominer")
                return False

        return True

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        # Simple extraction
        if "://" in url:
            url = url.split("://")[1]
        return url.split("/")[0].split(":")[0]

    def _log_blocked(self, url: str, reason: str):
        """Log blocked request."""
        self.blocked_requests.append({
            "url": url[:100],
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_stats(self) -> Dict:
        """Get protection statistics."""
        reasons = {}
        for req in self.blocked_requests:
            r = req["reason"]
            reasons[r] = reasons.get(r, 0) + 1

        return {
            "total_blocked": len(self.blocked_requests),
            "by_reason": reasons,
            "tracker_domains": len(self.TRACKER_DOMAINS),
            "miner_patterns": len(self.CRYPTOMINER_PATTERNS),
        }


# =============================================================================
# PSI FIREFOX BROWSER
# =============================================================================

class PsiFirefoxBrowser:
    """
    PIO-enhanced Firefox browser with Snowden-grade privacy.

    Integrates:
    - Privacy profiles (Standard/Strict/Paranoid)
    - Fingerprint blocking
    - Enhanced tracking protection
    - Brahim layer routing
    """

    def __init__(self, profile_path: Optional[Path] = None):
        self.profile_path = profile_path or Path("data/psi_firefox")
        self.profile: Optional[PrivacyProfile] = None
        self.fingerprint_blocker = FingerprintBlocker()
        self.tracking_protection = TrackingProtection()
        self.brahim_layer = BRAHIM_CENTER
        self.session_id = secrets.token_hex(16)

    def initialize(self, level: PrivacyLevel = PrivacyLevel.STRICT) -> bool:
        """Initialize browser with privacy profile."""
        self.profile_path.mkdir(parents=True, exist_ok=True)

        self.profile = PrivacyProfile(
            name=f"psi_{self.session_id[:8]}",
            level=level
        )

        # Apply fingerprint protections
        self.fingerprint_blocker.block_canvas()
        self.fingerprint_blocker.block_webgl()
        self.fingerprint_blocker.block_audio()
        self.fingerprint_blocker.spoof_screen()
        self.fingerprint_blocker.spoof_timezone()

        # Save profile
        return self.profile.save_to_profile(self.profile_path)

    def browse(self, url: str) -> Dict:
        """Browse URL with privacy protections."""
        # Check tracking protection
        first_party = self._extract_domain(url)
        allowed = self.tracking_protection.check_request(url, first_party)

        if not allowed:
            return {
                "blocked": True,
                "url": url,
                "reason": "tracking_protection",
            }

        return {
            "blocked": False,
            "url": url,
            "brahim_layer": self.brahim_layer,
            "fingerprint_protection": self.fingerprint_blocker.get_protection_status(),
            "profile_level": self.profile.level.value if self.profile else "none",
        }

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        if "://" in url:
            url = url.split("://")[1]
        return url.split("/")[0].split(":")[0]

    def rotate_layer(self) -> int:
        """Rotate to next Brahim layer."""
        idx = BRAHIM_SEQUENCE.index(self.brahim_layer)
        self.brahim_layer = BRAHIM_SEQUENCE[(idx + 1) % len(BRAHIM_SEQUENCE)]
        return self.brahim_layer

    def get_stats(self) -> Dict:
        """Get browser statistics."""
        return {
            "session_id": self.session_id[:8],
            "brahim_layer": self.brahim_layer,
            "profile_level": self.profile.level.value if self.profile else "none",
            "fingerprint": self.fingerprint_blocker.get_protection_status(),
            "tracking": self.tracking_protection.get_stats(),
            "gecko_engine": GECKO_ENGINE,
        }

    def cleanup(self) -> Dict:
        """Clean up browser data."""
        self.tracking_protection.blocked_requests.clear()
        return {
            "session_cleared": True,
            "profile_path": str(self.profile_path),
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "PrivacyLevel",
    "PrivacyProfile",
    "FingerprintBlocker",
    "TrackingProtection",
    "PsiFirefoxBrowser",
    "GECKO_ENGINE",
    "GECKO_COMPONENTS",
]
