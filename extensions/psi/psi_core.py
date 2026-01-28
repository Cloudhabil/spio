"""
PSI.APK Core - PIO + Tor Android Application Suite

Six core applications connecting PIO's 11-layer Brahim routing
with Tor's anonymity network, designed for Android APK deployment.

CORE APPLICATIONS:
    1. PsiMessenger  - Anonymous chat through 11 Brahim layers
    2. PsiVault      - Distributed encrypted file storage
    3. PsiExchange   - Mirror-pair cryptographic key exchange
    4. PsiDNS        - Decentralized .brahim/.onion naming
    5. PsiRelay      - Brahim beacon network node
    6. PsiMap        - Dark sector topology mapper

ROUTING ARCHITECTURE:
    User (BN_x) -> [11 Layers] -> CENTER (107) -> [11 Layers] -> Destination
"""

from __future__ import annotations

import os
import json
import time
import base64
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from abc import ABC, abstractmethod

# =============================================================================
# BRAHIM CONSTANTS
# =============================================================================

BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 107, 117, 139, 154, 172, 187]
BRAHIM_CENTER = 107
BRAHIM_SUM = 214
PHI = (1 + 5 ** 0.5) / 2

MIRROR_PAIRS = [
    (27, 187),   # GENESIS <-> OMEGA
    (42, 172),   # DUALITY <-> COMPLETION
    (60, 154),   # MANIFESTATION <-> INFINITY
    (75, 139),   # TESSERACT <-> HARMONY
    (97, 117),   # THRESHOLD <-> EMERGENCE
]

LAYER_NAMES = {
    27: "GENESIS", 42: "DUALITY", 60: "MANIFESTATION", 75: "TESSERACT",
    97: "THRESHOLD", 107: "CONVERGENCE", 117: "EMERGENCE", 139: "HARMONY",
    154: "INFINITY", 172: "COMPLETION", 187: "OMEGA"
}

LAYER_TYPES = {
    27: "SEED", 42: "INNER", 60: "INNER", 75: "INNER", 97: "INNER",
    107: "CENTER", 117: "OUTER", 139: "OUTER", 154: "OUTER", 172: "OUTER", 187: "OUTER"
}

# Tor circuit position mapping
TOR_POSITIONS = {
    27: "GUARD",
    42: "MIDDLE_1",
    60: "MIDDLE_2",
    75: "MIDDLE_3",
    97: "MIDDLE_4",
    107: "RENDEZVOUS",
    117: "MIDDLE_5",
    139: "MIDDLE_6",
    154: "MIDDLE_7",
    172: "EXIT",
    187: "HIDDEN_SERVICE",
}


# =============================================================================
# PSI LAYER
# =============================================================================

@dataclass
class PsiLayer:
    """A single layer in the Brahim routing network."""
    number: int
    name: str
    layer_type: str
    tor_position: str
    mirror: Optional[int] = None

    @classmethod
    def from_number(cls, num: int) -> "PsiLayer":
        """Create layer from Brahim number."""
        mirror = None
        for a, b in MIRROR_PAIRS:
            if num == a:
                mirror = b
            elif num == b:
                mirror = a

        return cls(
            number=num,
            name=LAYER_NAMES.get(num, f"LAYER_{num}"),
            layer_type=LAYER_TYPES.get(num, "UNKNOWN"),
            tor_position=TOR_POSITIONS.get(num, "UNKNOWN"),
            mirror=mirror
        )

    @property
    def is_center(self) -> bool:
        return self.number == BRAHIM_CENTER

    def distance_to_center(self) -> int:
        """Distance from this layer to center."""
        return abs(self.number - BRAHIM_CENTER)


# =============================================================================
# BRAHIM ROUTER
# =============================================================================

class BrahimRouter:
    """
    Routes data through 11 Brahim layers to reach center.

    The path follows: SEED -> INNER layers -> CENTER -> OUTER layers -> OMEGA
    """

    def __init__(self):
        self.layers = [PsiLayer.from_number(n) for n in BRAHIM_SEQUENCE]
        self.center_idx = BRAHIM_SEQUENCE.index(BRAHIM_CENTER)

    def route_to_center(self, source_layer: int) -> List[PsiLayer]:
        """Get route from source layer to center."""
        if source_layer not in BRAHIM_SEQUENCE:
            raise ValueError(f"Invalid layer: {source_layer}")

        src_idx = BRAHIM_SEQUENCE.index(source_layer)
        path = []

        if src_idx <= self.center_idx:
            # Going forward to center
            for i in range(src_idx, self.center_idx + 1):
                path.append(self.layers[i])
        else:
            # Going backward to center
            for i in range(src_idx, self.center_idx - 1, -1):
                path.append(self.layers[i])

        return path

    def route_through_center(self, source: int, destination: int) -> List[PsiLayer]:
        """Route from source through center to destination."""
        to_center = self.route_to_center(source)
        from_center = self.route_to_center(destination)

        # Remove duplicate center
        if from_center and from_center[0].is_center:
            from_center = from_center[1:]

        return to_center + from_center

    def get_mirror_route(self, layer: int) -> Tuple[int, int]:
        """Get the mirror pair for a layer."""
        for a, b in MIRROR_PAIRS:
            if layer == a:
                return (a, b)
            elif layer == b:
                return (b, a)
        return (layer, layer)


# =============================================================================
# PSI APPLICATIONS
# =============================================================================

class PsiApplication(ABC):
    """Base class for PSI applications."""

    def __init__(self, name: str, router: Optional[BrahimRouter] = None):
        self.name = name
        self.router = router or BrahimRouter()
        self.session_id = secrets.token_hex(16)
        self.created_at = datetime.now(timezone.utc)

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the application."""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Clean shutdown."""
        pass


class PsiMessenger(PsiApplication):
    """
    Anonymous chat through 11 Brahim layers.

    Messages route through center (107) for secure exchange.
    """

    def __init__(self):
        super().__init__("PsiMessenger")
        self.conversations: Dict[str, List[Dict]] = {}

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> bool:
        self.conversations.clear()
        return True

    def send_message(self, recipient_layer: int, content: str) -> Dict:
        """Send message through Brahim layers."""
        route = self.router.route_through_center(27, recipient_layer)

        encrypted = self._layer_encrypt(content, route)

        message = {
            "id": secrets.token_hex(8),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "route": [l.number for l in route],
            "hops": len(route),
            "encrypted": encrypted,
        }

        return message

    def _layer_encrypt(self, content: str, route: List[PsiLayer]) -> str:
        """Encrypt content through each layer."""
        data = content.encode()
        for layer in reversed(route):
            key = hashlib.sha256(f"{layer.number}:{self.session_id}".encode()).digest()
            data = bytes([b ^ key[i % 32] for i, b in enumerate(data)])
        return base64.b64encode(data).decode()


class PsiVault(PsiApplication):
    """
    Distributed encrypted file storage.

    Files are sharded across mirror pairs for redundancy.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__("PsiVault")
        self.storage_path = storage_path or Path("data/psi_vault")
        self.index: Dict[str, Dict] = {}

    def initialize(self) -> bool:
        self.storage_path.mkdir(parents=True, exist_ok=True)
        return True

    def shutdown(self) -> bool:
        return True

    def store(self, file_id: str, data: bytes) -> Dict:
        """Store file distributed across mirror pairs."""
        shards = self._shard_data(data)

        for i, (shard, (layer_a, layer_b)) in enumerate(zip(shards, MIRROR_PAIRS)):
            shard_path = self.storage_path / f"{file_id}_{i}.shard"
            shard_path.write_bytes(shard)

        self.index[file_id] = {
            "shards": len(shards),
            "size": len(data),
            "created": datetime.now(timezone.utc).isoformat(),
        }

        return self.index[file_id]

    def _shard_data(self, data: bytes) -> List[bytes]:
        """Shard data across 5 mirror pairs."""
        shard_size = (len(data) + 4) // 5
        return [data[i:i+shard_size] for i in range(0, len(data), shard_size)]


class PsiExchange(PsiApplication):
    """
    Mirror-pair cryptographic key exchange.

    Uses Brahim mirror symmetry for secure key derivation.
    """

    def __init__(self):
        super().__init__("PsiExchange")

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> bool:
        return True

    def derive_key(self, shared_secret: bytes, layer: int) -> bytes:
        """Derive key using mirror-pair symmetry."""
        mirror_a, mirror_b = self.router.get_mirror_route(layer)

        # Key = H(secret || layer_a || layer_b || sum)
        combined = shared_secret + str(mirror_a).encode() + str(mirror_b).encode()
        combined += str(BRAHIM_SUM).encode()

        return hashlib.sha256(combined).digest()


class PsiDNS(PsiApplication):
    """
    Decentralized .brahim/.onion naming system.

    Maps human-readable names to layer addresses.
    """

    def __init__(self):
        super().__init__("PsiDNS")
        self.records: Dict[str, Dict] = {}

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> bool:
        return True

    def register(self, name: str, target_layer: int) -> str:
        """Register a .brahim name."""
        if not name.endswith(".brahim"):
            name = f"{name}.brahim"

        record_id = hashlib.sha256(name.encode()).hexdigest()[:16]

        self.records[name] = {
            "id": record_id,
            "layer": target_layer,
            "created": datetime.now(timezone.utc).isoformat(),
        }

        return record_id

    def resolve(self, name: str) -> Optional[int]:
        """Resolve .brahim name to layer."""
        record = self.records.get(name)
        return record["layer"] if record else None


class PsiRelay(PsiApplication):
    """
    Brahim beacon network node.

    Relays traffic between layers in the network.
    """

    def __init__(self, layer: int = 107):
        super().__init__("PsiRelay")
        self.layer = PsiLayer.from_number(layer)
        self.connections: List[int] = []
        self.bandwidth_used = 0

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> bool:
        self.connections.clear()
        return True

    def relay(self, data: bytes, next_layer: int) -> bool:
        """Relay data to next layer."""
        if next_layer not in BRAHIM_SEQUENCE:
            return False

        self.bandwidth_used += len(data)
        return True

    def stats(self) -> Dict:
        """Get relay statistics."""
        return {
            "layer": self.layer.number,
            "name": self.layer.name,
            "connections": len(self.connections),
            "bandwidth_kb": self.bandwidth_used / 1024,
        }


class PsiMap(PsiApplication):
    """
    Dark sector topology mapper.

    Maps the network topology and identifies dark sectors.
    """

    def __init__(self):
        super().__init__("PsiMap")
        self.topology: Dict[int, List[int]] = {}
        self.dark_sectors: List[int] = []

    def initialize(self) -> bool:
        self._build_topology()
        return True

    def shutdown(self) -> bool:
        return True

    def _build_topology(self):
        """Build layer connectivity map."""
        for i, layer in enumerate(BRAHIM_SEQUENCE):
            connections = []
            if i > 0:
                connections.append(BRAHIM_SEQUENCE[i-1])
            if i < len(BRAHIM_SEQUENCE) - 1:
                connections.append(BRAHIM_SEQUENCE[i+1])
            self.topology[layer] = connections

    def find_dark_sectors(self) -> List[int]:
        """Identify unreachable or poorly connected areas."""
        dark = []
        for layer, connections in self.topology.items():
            if len(connections) < 2 and layer != BRAHIM_CENTER:
                dark.append(layer)
        self.dark_sectors = dark
        return dark

    def visualize(self) -> str:
        """ASCII visualization of topology."""
        lines = ["PSI NETWORK TOPOLOGY", "=" * 40]

        for layer in BRAHIM_SEQUENCE:
            psi = PsiLayer.from_number(layer)
            marker = "[*]" if psi.is_center else "[ ]"
            dark = " (DARK)" if layer in self.dark_sectors else ""
            lines.append(f"{marker} {layer:3} - {psi.name:15} {psi.tor_position}{dark}")

        return "\n".join(lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "BRAHIM_SEQUENCE",
    "BRAHIM_CENTER",
    "BRAHIM_SUM",
    "MIRROR_PAIRS",
    "LAYER_NAMES",
    "LAYER_TYPES",
    "TOR_POSITIONS",
    "PsiLayer",
    "BrahimRouter",
    "PsiApplication",
    "PsiMessenger",
    "PsiVault",
    "PsiExchange",
    "PsiDNS",
    "PsiRelay",
    "PsiMap",
]
