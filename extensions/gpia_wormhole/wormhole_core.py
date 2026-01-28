"""
GPIA Wormhole Core - Unified Wormhole Routing Implementation

8 Components for instant information retrieval:
1. PacketGraph - Packet-switched network structure
2. DataPacket - Routable data unit with Brahim address
3. WormholeTopology - Hyperbolic geometry (Poincare disk)
4. WormholeRouter - Query routing with safety
5. CharacteristicSolver - PDE method of characteristics
6. ResonanceTransport - Resonance along characteristics
7. CharacteristicManifold - Complete wormhole manifold
8. HighDimCharacteristics - 384-D embedding geodesics
"""

from __future__ import annotations

import math
import hashlib
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum
import numpy as np


# =============================================================================
# BRAHIM CONSTANTS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2        # 1.6180339887498949
BETA = math.sqrt(5) - 2             # 0.2360679774997897
ALPHA = PHI - 1                      # 0.6180339887498949
GENESIS = 0.0219                     # Axiological alignment constant
LAMBDA_DECAY = GENESIS               # Decay rate along characteristics

BRAHIM_SEQUENCE = [27, 42, 60, 75, 97, 117, 139, 154, 172, 187]
BRAHIM_SUM = 214
CURVATURE = -4 * BETA ** 2           # -0.223 (hyperbolic)
EPSILON = 1e-10


# =============================================================================
# COMPONENT 1: DATA PACKET
# =============================================================================

@dataclass
class DataPacket:
    """
    A data point treated as a routable packet.

    Each packet has:
    - Unique address (Brahim-encoded)
    - Payload (the actual data)
    - Embedding (for similarity routing)
    - Wormhole links (instant connections)
    """

    packet_id: str
    address: str = ""
    payload: Any = None
    embedding: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)
    source: str = "unknown"
    packet_type: str = "data"
    hop_count: int = 0
    ttl: int = 64
    priority: float = 1.0
    wormhole_links: Set[str] = field(default_factory=set)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
        if not self.address:
            self.address = self._encode_address()

    def _compute_hash(self) -> str:
        """Compute Brahim-weighted content hash."""
        content = str(self.payload).encode()
        base_hash = hashlib.sha256(content).hexdigest()

        weighted = ""
        for i, char in enumerate(base_hash[:8]):
            weight = BRAHIM_SEQUENCE[i % 10]
            weighted += format((int(char, 16) * weight) % 16, 'x')

        return weighted + base_hash[8:16]

    def _encode_address(self) -> str:
        """Generate Brahim-encoded address."""
        type_code = self.packet_type[0].upper()
        hash_segment = self.content_hash[:6]
        hash_int = int(hash_segment, 16)
        seq_index = hash_int % 10

        return f"B{type_code}:{seq_index}:{hash_segment}"

    def add_wormhole(self, target_id: str) -> None:
        """Add a wormhole link to another packet."""
        self.wormhole_links.add(target_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize packet to dictionary."""
        return {
            "packet_id": self.packet_id,
            "address": self.address,
            "payload": self.payload,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at,
            "source": self.source,
            "packet_type": self.packet_type,
            "wormhole_links": list(self.wormhole_links),
            "content_hash": self.content_hash,
        }


# =============================================================================
# COMPONENT 2: PACKET GRAPH
# =============================================================================

class PacketGraph:
    """
    Graph structure where all data points are packets.

    Features:
    - O(1) packet lookup by ID or address
    - Wormhole connections for instant traversal
    - Similarity-based routing
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._lock = threading.RLock()

        self._packets: Dict[str, DataPacket] = {}
        self._address_map: Dict[str, str] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)
        self._wormholes: Dict[str, Set[str]] = defaultdict(set)
        self._embeddings: List[Tuple[str, np.ndarray]] = []

        self._stats = {
            "packets_added": 0,
            "wormholes_created": 0,
            "queries_processed": 0,
            "wormhole_traversals": 0,
        }

    def add_packet(
        self,
        packet_id: str,
        payload: Any,
        embedding: Optional[np.ndarray] = None,
        packet_type: str = "data",
        source: str = "unknown",
        auto_wormhole: bool = True,
    ) -> DataPacket:
        """Add a packet to the graph."""
        with self._lock:
            packet = DataPacket(
                packet_id=packet_id,
                payload=payload,
                embedding=embedding,
                source=source,
                packet_type=packet_type,
            )

            self._packets[packet_id] = packet
            self._address_map[packet.address] = packet_id

            if embedding is not None:
                self._embeddings.append((packet_id, embedding))
                if auto_wormhole:
                    self._create_auto_wormholes(packet)

            self._stats["packets_added"] += 1
            return packet

    def _create_auto_wormholes(
        self,
        packet: DataPacket,
        threshold: float = ALPHA,
        max_wormholes: int = 5,
    ) -> None:
        """Automatically create wormholes to similar packets."""
        if packet.embedding is None:
            return

        similarities = []
        for other_id, other_emb in self._embeddings:
            if other_id == packet.packet_id:
                continue

            sim = self._cosine_similarity(packet.embedding, other_emb)
            if sim > threshold:
                similarities.append((other_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        for target_id, sim in similarities[:max_wormholes]:
            self._create_wormhole(packet.packet_id, target_id, sim)

    def _create_wormhole(self, source_id: str, target_id: str, strength: float = 1.0) -> None:
        """Create a bidirectional wormhole connection."""
        with self._lock:
            self._wormholes[source_id].add(target_id)
            self._wormholes[target_id].add(source_id)

            if source_id in self._packets:
                self._packets[source_id].add_wormhole(target_id)
            if target_id in self._packets:
                self._packets[target_id].add_wormhole(source_id)

            self._stats["wormholes_created"] += 1

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_packet(self, packet_id: str) -> Optional[DataPacket]:
        """Get packet by ID."""
        with self._lock:
            return self._packets.get(packet_id)

    def query_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        use_wormholes: bool = True,
    ) -> List[Tuple[DataPacket, float]]:
        """Find similar packets using wormhole-accelerated search."""
        with self._lock:
            self._stats["queries_processed"] += 1

            results = []
            for packet_id, embedding in self._embeddings:
                sim = self._cosine_similarity(query_embedding, embedding)
                results.append((packet_id, sim))

            results.sort(key=lambda x: x[1], reverse=True)

            if use_wormholes:
                results = self._expand_via_wormholes(results[:k])

            return [
                (self._packets[pid], sim)
                for pid, sim in results[:k]
                if pid in self._packets
            ]

    def _expand_via_wormholes(
        self,
        initial_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Expand results by traversing wormhole connections."""
        seen = set()
        expanded = list(initial_results)

        for packet_id, base_sim in initial_results[:3]:
            if packet_id not in self._wormholes:
                continue

            for wormhole_target in self._wormholes[packet_id]:
                if wormhole_target in seen:
                    continue
                seen.add(wormhole_target)

                wormhole_sim = base_sim * (1 + BETA)
                expanded.append((wormhole_target, min(wormhole_sim, 1.0)))
                self._stats["wormhole_traversals"] += 1

        expanded_dict = {}
        for pid, sim in expanded:
            if pid not in expanded_dict or sim > expanded_dict[pid]:
                expanded_dict[pid] = sim

        return sorted(expanded_dict.items(), key=lambda x: x[1], reverse=True)

    def stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._lock:
            return {
                **self._stats,
                "total_packets": len(self._packets),
                "total_wormholes": sum(len(w) for w in self._wormholes.values()) // 2,
            }


# =============================================================================
# COMPONENT 3: WORMHOLE TOPOLOGY
# =============================================================================

@dataclass
class HyperbolicPoint:
    """Point in Poincare disk model."""
    coords: np.ndarray
    radius: float = 0.0
    angle: float = 0.0

    def __post_init__(self):
        if len(self.coords) >= 2:
            self.radius = np.linalg.norm(self.coords[:2])
            self.angle = math.atan2(self.coords[1], self.coords[0]) if self.radius > 0 else 0

    @property
    def curvature_factor(self) -> float:
        """Local curvature factor (increases toward edge)."""
        return 1 / (1 - self.radius ** 2 + EPSILON)

    @property
    def wormhole_density(self) -> float:
        """Expected wormhole density at this point."""
        return self.curvature_factor * BETA


@dataclass
class WormholeConnection:
    """A wormhole in the topology."""
    source: str
    target: str
    strength: float
    geodesic_length: float
    traversal_cost: float

    @property
    def compression_ratio(self) -> float:
        """How much shorter than Euclidean path."""
        return self.traversal_cost / (self.geodesic_length + EPSILON)


class WormholeTopology:
    """Manages the geometric structure of the wormhole manifold."""

    def __init__(self, dimension: int = 384, disk_radius: float = 1.0):
        self.dimension = dimension
        self.disk_radius = disk_radius
        self.curvature = CURVATURE

        self._wormholes: Dict[str, WormholeConnection] = {}
        self._points: Dict[str, HyperbolicPoint] = {}
        self._stats = {
            "total_wormholes": 0,
            "avg_geodesic_length": 0.0,
            "topology_updates": 0,
        }

    def project_to_disk(self, embedding: np.ndarray) -> HyperbolicPoint:
        """Project high-dimensional embedding to Poincare disk."""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return HyperbolicPoint(coords=np.zeros(self.dimension))

        disk_radius = math.tanh(norm * BETA)
        coords = (embedding / norm) * disk_radius

        return HyperbolicPoint(coords=coords)

    def hyperbolic_distance(self, p1: HyperbolicPoint, p2: HyperbolicPoint) -> float:
        """Compute hyperbolic distance between two points."""
        diff = p1.coords - p2.coords
        diff_norm_sq = np.dot(diff, diff)

        p1_norm_sq = np.dot(p1.coords, p1.coords)
        p2_norm_sq = np.dot(p2.coords, p2.coords)

        denom = (1 - p1_norm_sq) * (1 - p2_norm_sq)
        if denom <= 0:
            denom = EPSILON

        inner = 1 + 2 * diff_norm_sq / denom
        inner = max(inner, 1.0)

        return math.acosh(inner)

    def register_point(self, packet_id: str, embedding: np.ndarray) -> HyperbolicPoint:
        """Register a packet's position in the topology."""
        point = self.project_to_disk(embedding)
        self._points[packet_id] = point
        return point

    def add_wormhole(
        self,
        source_id: str,
        target_id: str,
        strength: float = 1.0,
    ) -> Optional[WormholeConnection]:
        """Add a wormhole connection."""
        if source_id not in self._points or target_id not in self._points:
            return None

        p1 = self._points[source_id]
        p2 = self._points[target_id]

        geodesic = self.hyperbolic_distance(p1, p2)
        traversal_cost = geodesic * BETA

        wormhole = WormholeConnection(
            source=source_id,
            target=target_id,
            strength=strength,
            geodesic_length=geodesic,
            traversal_cost=traversal_cost,
        )

        key = f"{source_id}:{target_id}"
        self._wormholes[key] = wormhole
        self._stats["total_wormholes"] += 1

        return wormhole

    def stats(self) -> Dict[str, Any]:
        """Get topology statistics."""
        return {
            **self._stats,
            "registered_points": len(self._points),
            "curvature": self.curvature,
        }


# =============================================================================
# COMPONENT 4: WORMHOLE ROUTER
# =============================================================================

class RouteType(Enum):
    """Type of routing path."""
    DIRECT = "direct"
    BRIDGE = "bridge"
    CASCADE = "cascade"
    STANDARD = "standard"
    BLOCKED = "blocked"


class SafetyLevel(Enum):
    """ASIOS safety classification."""
    SAFE = "safe"
    NOMINAL = "nominal"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


@dataclass
class RoutingResult:
    """Result of a wormhole routing operation."""
    query: str
    packets: List[DataPacket]
    route_type: RouteType
    hops: int
    latency_ms: float
    safety_level: SafetyLevel
    wormholes_used: int
    path: List[str]
    similarity_scores: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.packets) > 0 and self.safety_level != SafetyLevel.BLOCKED


class WormholeRouter:
    """Routes queries through the packet manifold using wormhole shortcuts."""

    def __init__(
        self,
        graph: PacketGraph,
        safety_checker: Optional[Callable[[str], SafetyLevel]] = None,
    ):
        self.graph = graph
        self.safety_checker = safety_checker or self._default_safety_check
        self._cache: Dict[str, Tuple[RoutingResult, float]] = {}
        self._stats = {"total_queries": 0, "cache_hits": 0}

    def _default_safety_check(self, query: str) -> SafetyLevel:
        """Default ASIOS safety check."""
        unsafe_patterns = ["hack", "exploit", "bypass", "inject"]
        query_lower = query.lower()

        for pattern in unsafe_patterns:
            if pattern in query_lower:
                return SafetyLevel.CAUTION

        return SafetyLevel.SAFE

    def route(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        k: int = 5,
        use_cache: bool = True,
    ) -> RoutingResult:
        """Route a query through the packet manifold."""
        start_time = time.time()
        self._stats["total_queries"] += 1

        cache_key = f"{query}:{k}"
        if use_cache and cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if time.time() - cached_time < 60:
                self._stats["cache_hits"] += 1
                return cached_result

        safety_level = self.safety_checker(query)
        if safety_level == SafetyLevel.BLOCKED:
            return RoutingResult(
                query=query,
                packets=[],
                route_type=RouteType.BLOCKED,
                hops=0,
                latency_ms=(time.time() - start_time) * 1000,
                safety_level=SafetyLevel.BLOCKED,
                wormholes_used=0,
                path=[],
                similarity_scores=[],
            )

        # Query similar packets
        if query_embedding is not None:
            results = self.graph.query_similar(query_embedding, k=k)
        else:
            results = []

        packets = [p for p, s in results]
        scores = [s for p, s in results]
        wormholes_used = self.graph._stats.get("wormhole_traversals", 0)

        route_type = RouteType.DIRECT if wormholes_used > 0 else RouteType.STANDARD

        result = RoutingResult(
            query=query,
            packets=packets,
            route_type=route_type,
            hops=len(packets),
            latency_ms=(time.time() - start_time) * 1000,
            safety_level=safety_level,
            wormholes_used=wormholes_used,
            path=[p.packet_id for p in packets],
            similarity_scores=scores,
        )

        self._cache[cache_key] = (result, time.time())
        return result


# =============================================================================
# COMPONENT 5: CHARACTERISTIC SOLVER
# =============================================================================

class PDEType(Enum):
    """Classification of PDEs by characteristic type."""
    HYPERBOLIC = "hyperbolic"
    PARABOLIC = "parabolic"
    ELLIPTIC = "elliptic"
    UNKNOWN = "unknown"


@dataclass
class CharacteristicPoint:
    """A point along a characteristic curve."""
    s: float
    x: float
    y: float
    u: float
    p: float = 0.0
    q: float = 0.0

    def distance_to(self, other: 'CharacteristicPoint') -> float:
        """Euclidean distance to another point."""
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.u - other.u)**2
        )

    def wormhole_distance_to(self, other: 'CharacteristicPoint') -> float:
        """Wormhole-compressed distance."""
        return self.distance_to(other) * BETA


@dataclass
class Characteristic:
    """A complete characteristic curve."""
    id: str
    family: int
    points: List[CharacteristicPoint]
    pde_type: PDEType = PDEType.HYPERBOLIC
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        """Arc length of characteristic."""
        if len(self.points) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.points)):
            total += self.points[i-1].distance_to(self.points[i])
        return total


@dataclass
class WormholeJump:
    """A wormhole connection between two characteristic families."""
    source_char_id: str
    target_char_id: str
    source_point: CharacteristicPoint
    target_point: CharacteristicPoint
    euclidean_distance: float
    wormhole_distance: float
    strength: float

    @property
    def compression_ratio(self) -> float:
        return self.wormhole_distance / (self.euclidean_distance + EPSILON)


class CharacteristicSolver:
    """
    Solves characteristic equations for first-order PDEs.

    Given PDE: a(x,y,u)*ux + b(x,y,u)*uy = c(x,y,u)

    Characteristic equations:
        dx/ds = a(x,y,u)
        dy/ds = b(x,y,u)
        du/ds = c(x,y,u)
    """

    def __init__(
        self,
        a: Callable[[float, float, float], float],
        b: Callable[[float, float, float], float],
        c: Callable[[float, float, float], float],
        tol: float = 1e-6
    ):
        self.a = a
        self.b = b
        self.c = c
        self.tol = tol

    def _rk4_step(self, state: np.ndarray, ds: float) -> np.ndarray:
        """Single RK4 step for characteristic equations."""
        x, y, u = state

        def deriv(st):
            return np.array([
                self.a(st[0], st[1], st[2]),
                self.b(st[0], st[1], st[2]),
                self.c(st[0], st[1], st[2])
            ])

        k1 = deriv(state)
        k2 = deriv(state + ds*k1/2)
        k3 = deriv(state + ds*k2/2)
        k4 = deriv(state + ds*k3)

        return state + (ds/6) * (k1 + 2*k2 + 2*k3 + k4)

    def solve(
        self,
        x0: float,
        y0: float,
        u0: float,
        s_range: Tuple[float, float],
        ds: float = 0.01,
        max_steps: int = 10000
    ) -> Characteristic:
        """Solve characteristic curve from initial point."""
        s_min, s_max = s_range
        points = []

        state = np.array([x0, y0, u0])
        s = s_min

        points.append(CharacteristicPoint(s=s, x=x0, y=y0, u=u0))

        step_count = 0
        while s < s_max and step_count < max_steps:
            state = self._rk4_step(state, ds)
            s += ds

            points.append(CharacteristicPoint(
                s=s,
                x=state[0],
                y=state[1],
                u=state[2]
            ))

            step_count += 1

        return Characteristic(
            id=f"char_{id(points)}",
            family=0,
            points=points,
            metadata={"steps": step_count, "final_s": s}
        )


# =============================================================================
# COMPONENT 6: RESONANCE TRANSPORT
# =============================================================================

class ResonanceTransport:
    """
    Transport of resonance along characteristics in the 4D Ball Tree manifold.

    The resonance equation from ASI-OS:
        R(t) = sum(1/(||vi - q||^2 + eps)) * e^(-lambda*(t-ti))
    """

    def __init__(self, lambda_decay: float = LAMBDA_DECAY):
        self.lambda_decay = lambda_decay
        self._memory_points: List[Tuple[np.ndarray, float]] = []
        self._lock = threading.Lock()

    def add_memory(self, vector: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Add a memory point to the manifold."""
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self._memory_points.append((vector.copy(), timestamp))

    def source_term(self, query: np.ndarray, t: float) -> float:
        """Compute source term S(x,t)."""
        with self._lock:
            if not self._memory_points:
                return 0.0

            total = 0.0
            for vi, ti in self._memory_points:
                dist_sq = np.sum((vi - query) ** 2)
                spatial_term = 1.0 / (dist_sq + EPSILON)
                # Temporal decay only if t >= ti (memory from the past)
                delta_t = max(0, t - ti)
                temporal_term = math.exp(-self.lambda_decay * delta_t)
                total += spatial_term * temporal_term

            return total

    def solve_characteristic(
        self,
        query: np.ndarray,
        t_range: Tuple[float, float],
        R0: float = 0.0,
        dt: float = 0.01
    ) -> List[Tuple[float, float]]:
        """Solve resonance transport along time characteristic."""
        t_min, t_max = t_range
        path = [(t_min, R0)]

        R = R0
        t = t_min

        while t < t_max:
            S = self.source_term(query, t)
            dR = -self.lambda_decay * R + S
            R += dR * dt
            t += dt
            path.append((t, R))

        return path

    def compute_axiological_alignment(self, R_observed: float) -> float:
        """Compute delta_sys = |R_obs - 0.0219|."""
        return abs(R_observed - LAMBDA_DECAY)


# =============================================================================
# COMPONENT 7: CHARACTERISTIC MANIFOLD
# =============================================================================

class CharacteristicManifold:
    """
    Complete manifold where characteristics define optimal routing paths.

    Key concepts:
    - Characteristics = geodesics / natural information flow
    - Wormholes = discontinuous jumps between characteristic families
    - Resonance propagates along characteristics with decay lambda
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._families: Dict[int, List[Characteristic]] = {}
        self._family_count = 0
        self._wormholes: List[WormholeJump] = []
        self.resonance = ResonanceTransport(LAMBDA_DECAY)
        self._stats = {
            "characteristics_computed": 0,
            "wormholes_discovered": 0,
            "total_arc_length": 0.0,
        }
        self._lock = threading.Lock()

    def add_characteristic(
        self,
        characteristic: Characteristic,
        family: Optional[int] = None
    ) -> int:
        """Add a characteristic to the manifold."""
        with self._lock:
            if family is None:
                family = self._family_count
                self._family_count += 1

            characteristic.family = family

            if family not in self._families:
                self._families[family] = []

            self._families[family].append(characteristic)
            self._stats["characteristics_computed"] += 1
            self._stats["total_arc_length"] += characteristic.length

            return family

    def compute_transport_characteristics(
        self,
        initial_curve: List[Tuple[float, float]],
        velocity: float = 1.0,
        t_max: float = 10.0
    ) -> int:
        """Compute characteristics for simple transport equation."""
        solver = CharacteristicSolver(
            a=lambda x, t, u: velocity,
            b=lambda x, t, u: 1.0,
            c=lambda x, t, u: 0.0
        )

        family = self._family_count
        for x0, u0 in initial_curve:
            char = solver.solve(x0, 0.0, u0, (0, t_max))
            self.add_characteristic(char, family)

        return family

    def discover_wormholes(
        self,
        family1: int,
        family2: int,
        threshold: float = 1.0,
        max_wormholes: int = 10
    ) -> List[WormholeJump]:
        """Discover wormhole connections between two characteristic families."""
        with self._lock:
            if family1 not in self._families or family2 not in self._families:
                return []

            chars1 = self._families[family1]
            chars2 = self._families[family2]

        candidates = []

        for c1 in chars1:
            for c2 in chars2:
                for p1 in c1.points[::10]:
                    for p2 in c2.points[::10]:
                        dist = p1.distance_to(p2)
                        if dist < threshold:
                            wormhole_dist = dist * BETA
                            strength = 1.0 - (dist / threshold)

                            candidates.append(WormholeJump(
                                source_char_id=c1.id,
                                target_char_id=c2.id,
                                source_point=p1,
                                target_point=p2,
                                euclidean_distance=dist,
                                wormhole_distance=wormhole_dist,
                                strength=strength
                            ))

        candidates.sort(key=lambda w: w.strength, reverse=True)
        wormholes = candidates[:max_wormholes]

        with self._lock:
            self._wormholes.extend(wormholes)
            self._stats["wormholes_discovered"] += len(wormholes)

        return wormholes

    def get_stats(self) -> Dict[str, Any]:
        """Get manifold statistics."""
        with self._lock:
            return {
                **self._stats,
                "num_families": len(self._families),
                "num_wormholes": len(self._wormholes),
                "genesis_constant": LAMBDA_DECAY,
                "beta_compression": BETA,
            }


# =============================================================================
# COMPONENT 8: HIGH-DIMENSIONAL CHARACTERISTICS
# =============================================================================

class HighDimCharacteristics:
    """
    Characteristics in high-dimensional embedding space (384-D).

    In the Ball Tree manifold, characteristics are geodesics
    in the hyperbolic Poincare disk projection.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.manifold = CharacteristicManifold(dimension)

    def project_to_2d(self, embedding: np.ndarray) -> Tuple[float, float]:
        """Project high-D embedding to 2D for characteristic computation."""
        if len(embedding) < 2:
            return (0.0, 0.0)

        norm = np.linalg.norm(embedding)
        if norm < EPSILON:
            return (0.0, 0.0)

        r = math.tanh(norm * BETA)
        theta = math.atan2(embedding[1], embedding[0])

        return (r * math.cos(theta), r * math.sin(theta))

    def compute_embedding_characteristic(
        self,
        start_embedding: np.ndarray,
        direction: np.ndarray,
        num_steps: int = 100
    ) -> Characteristic:
        """Compute characteristic through embedding space."""
        points = []

        dir_norm = np.linalg.norm(direction)
        if dir_norm < EPSILON:
            direction = np.random.randn(self.dimension)
            dir_norm = np.linalg.norm(direction)
        direction = direction / dir_norm

        step_size = 0.1 * BETA
        current = start_embedding.copy()

        for i in range(num_steps):
            x, y = self.project_to_2d(current)
            u = np.linalg.norm(current)

            points.append(CharacteristicPoint(
                s=i * step_size,
                x=x,
                y=y,
                u=u
            ))

            current = current + step_size * direction
            norm = np.linalg.norm(current)
            if norm > 1 - EPSILON:
                current = current / norm * (1 - EPSILON)

        return Characteristic(
            id=f"emb_char_{id(points)}",
            family=0,
            points=points,
            metadata={"dimension": self.dimension}
        )

    def find_wormhole_in_embedding_space(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> WormholeJump:
        """Compute wormhole properties between two embeddings."""
        x1, y1 = self.project_to_2d(embedding1)
        x2, y2 = self.project_to_2d(embedding2)

        p1 = CharacteristicPoint(s=0, x=x1, y=y1, u=np.linalg.norm(embedding1))
        p2 = CharacteristicPoint(s=0, x=x2, y=y2, u=np.linalg.norm(embedding2))

        euclidean = np.linalg.norm(embedding1 - embedding2)
        wormhole = euclidean * BETA

        cos_sim = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + EPSILON
        )
        strength = max(0, cos_sim)

        return WormholeJump(
            source_char_id="embedding_1",
            target_char_id="embedding_2",
            source_point=p1,
            target_point=p2,
            euclidean_distance=euclidean,
            wormhole_distance=wormhole,
            strength=strength
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_transport_solver(velocity: float = 1.0) -> CharacteristicSolver:
    """Create solver for transport equation ut + c*ux = 0."""
    return CharacteristicSolver(
        a=lambda x, t, u: velocity,
        b=lambda x, t, u: 1.0,
        c=lambda x, t, u: 0.0
    )


def create_resonance_transport(lambda_decay: float = LAMBDA_DECAY) -> ResonanceTransport:
    """Create resonance transport with specified decay constant."""
    return ResonanceTransport(lambda_decay)


def create_characteristic_manifold(dimension: int = 384) -> CharacteristicManifold:
    """Create a characteristic manifold for wormhole routing."""
    return CharacteristicManifold(dimension)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "PHI", "BETA", "ALPHA", "GENESIS", "LAMBDA_DECAY", "CURVATURE",
    "BRAHIM_SEQUENCE", "BRAHIM_SUM",

    # Packet Graph
    "DataPacket", "PacketGraph",

    # Topology
    "HyperbolicPoint", "WormholeConnection", "WormholeTopology",

    # Router
    "RouteType", "SafetyLevel", "RoutingResult", "WormholeRouter",

    # Characteristics
    "PDEType", "CharacteristicPoint", "Characteristic", "WormholeJump",
    "CharacteristicSolver", "ResonanceTransport", "CharacteristicManifold",
    "HighDimCharacteristics",

    # Factories
    "create_transport_solver", "create_resonance_transport",
    "create_characteristic_manifold",
]
