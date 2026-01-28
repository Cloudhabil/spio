"""
GPIA Wormhole Extension - Instant Information Retrieval

Treats the manifold as a packet-switched network where:
- Each data point (voxel, embedding, skill) = Packet
- Wormhole connections = Instant traversal paths
- ASI-OS = Safety-aware routing governance

8 Core Components:
1. PacketGraph        - Data packets as routable network nodes
2. DataPacket         - Individual packet with Brahim address
3. WormholeTopology   - Hyperbolic geometry in Poincare disk
4. WormholeRouter     - Routes queries using wormhole shortcuts
5. CharacteristicSolver - PDE solving via method of characteristics
6. ResonanceTransport - Transport resonance along characteristics
7. CharacteristicManifold - Complete manifold for wormhole routing
8. HighDimCharacteristics - 384-D embedding space geodesics

Mathematical Foundation:
- Genesis Constant: lambda = 0.0219 (axiological alignment)
- Beta Compression: beta = sqrt(5) - 2 = 0.236 (wormhole factor)
- Curvature: K = -4*beta^2 = -0.223 (hyperbolic)

Author: Elias Oulad Brahim
Reference: ASI-OS Publication, Jan 2026
"""

from .wormhole_core import (
    # Constants
    PHI,
    BETA,
    ALPHA,
    GENESIS,
    LAMBDA_DECAY,
    CURVATURE,
    BRAHIM_SEQUENCE,
    BRAHIM_SUM,

    # Packet Graph
    DataPacket,
    PacketGraph,

    # Topology
    HyperbolicPoint,
    WormholeConnection,
    WormholeTopology,

    # Router
    RouteType,
    SafetyLevel,
    RoutingResult,
    WormholeRouter,

    # Characteristics
    PDEType,
    CharacteristicPoint,
    Characteristic,
    WormholeJump,
    CharacteristicSolver,
    ResonanceTransport,
    CharacteristicManifold,
    HighDimCharacteristics,

    # Factories
    create_transport_solver,
    create_resonance_transport,
    create_characteristic_manifold,
)

__all__ = [
    # Constants
    "PHI",
    "BETA",
    "ALPHA",
    "GENESIS",
    "LAMBDA_DECAY",
    "CURVATURE",
    "BRAHIM_SEQUENCE",
    "BRAHIM_SUM",

    # Packet Graph
    "DataPacket",
    "PacketGraph",

    # Topology
    "HyperbolicPoint",
    "WormholeConnection",
    "WormholeTopology",

    # Router
    "RouteType",
    "SafetyLevel",
    "RoutingResult",
    "WormholeRouter",

    # Characteristics
    "PDEType",
    "CharacteristicPoint",
    "Characteristic",
    "WormholeJump",
    "CharacteristicSolver",
    "ResonanceTransport",
    "CharacteristicManifold",
    "HighDimCharacteristics",

    # Factories
    "create_transport_solver",
    "create_resonance_transport",
    "create_characteristic_manifold",
]
