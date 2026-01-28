"""
PSI Extension - Privacy & Security Infrastructure

Three-layer privacy system integrating PIO with anonymity networks.

Layers:
1. PSI.APK - Android Tor integration (Orbot/Guardian Project)
2. PSI.Onion - Tor Browser wrapper with Snowden-grade skills
3. PSI.Firefox - Firefox privacy enhancement

BRAHIM ROUTING:
    User -> [11 Brahim Layers] -> CENTER (107) -> [11 Layers] -> Destination

INTEGRATED SKILLS (from Snowden corpus):
- SecureDrop: Air-gap, no-log, metadata stripping
- OnionShare: Ephemeral .onion, anonymous dropbox
- GlobaLeaks: 16-digit receipts, Argon2ID, auto-delete
- Briar: No central server, offline sync
- Tails: RAM wipe, stream isolation
- Whonix: IP leak protection, keystroke anonymization
"""

from .psi_core import (
    PsiLayer,
    BrahimRouter,
    PsiMessenger,
    PsiVault,
    PsiExchange,
    PsiDNS,
    PsiRelay,
    PsiMap,
    BRAHIM_SEQUENCE,
    BRAHIM_CENTER,
    MIRROR_PAIRS,
    LAYER_NAMES,
)

from .psi_onion import (
    PsiOnionBrowser,
    ReceiptSystem,
    MetadataStripper,
    StreamIsolator,
    SecureSession,
)

from .psi_firefox import (
    PsiFirefoxBrowser,
    PrivacyProfile,
    FingerprintBlocker,
    TrackingProtection,
)

__all__ = [
    # Core
    "PsiLayer",
    "BrahimRouter",
    "PsiMessenger",
    "PsiVault",
    "PsiExchange",
    "PsiDNS",
    "PsiRelay",
    "PsiMap",
    "BRAHIM_SEQUENCE",
    "BRAHIM_CENTER",
    "MIRROR_PAIRS",
    "LAYER_NAMES",
    # Onion
    "PsiOnionBrowser",
    "ReceiptSystem",
    "MetadataStripper",
    "StreamIsolator",
    "SecureSession",
    # Firefox
    "PsiFirefoxBrowser",
    "PrivacyProfile",
    "FingerprintBlocker",
    "TrackingProtection",
]
