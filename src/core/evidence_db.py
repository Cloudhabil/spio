"""
Evidence Database - Lightweight SQLite audit trail per SPIO cycle.

Schema follows the Universal Structure Evidence Database pattern
(validated via Goldbach research, 2026-01-29).

Each cycle emits:
- Constants used (golden hierarchy snapshot)
- Axioms verified (pre-conditions)
- Operations performed (input, output, silicon, latency)
- Summary statistics (cycle metrics)

Author: SPIO Framework via Brahim's Calculator
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
from dataclasses import dataclass

from sovereign_pio.constants import (
    BETA,
    GAMMA,
    OMEGA,
    PHI,
    TOTAL_STATES,
)

# Derived constants
BRAHIM_CENTER: int = 107
BRAHIM_SUM: int = 214
PHI_PI_GAP: float = (322 * math.pi - 1000) / 1000
CLOSURE_EXPONENT: float = 2 / 3


@dataclass
class OperationRecord:
    """A single SPIO operation record."""
    op_id: int
    input_hash: int
    output_hash: int
    silicon: str
    dimension: int
    latency_ms: float
    bandwidth_gbps: float
    verified: bool


class EvidenceDB:
    """
    Lightweight SQLite evidence database for SPIO cycles.

    Usage:
        db = EvidenceDB("cycle_001.db")
        db.open()
        db.record_constants()
        db.verify_axioms()
        db.record_operation(op)
        db.finalize()
        db.close()
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(self, path: str):
        self.path = path
        self.conn: sqlite3.Connection | None = None
        self._op_count = 0
        self._start_time = 0.0

    def open(self) -> None:
        """Open database and create schema."""
        if os.path.exists(self.path):
            os.remove(self.path)
        self.conn = sqlite3.connect(self.path)
        self._create_schema()
        self._start_time = time.time()

    def _create_schema(self) -> None:
        assert self.conn is not None
        self.conn.executescript("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE constants (
                name TEXT PRIMARY KEY,
                value REAL,
                description TEXT
            );
            CREATE TABLE axioms (
                id TEXT PRIMARY KEY,
                name TEXT,
                verified INTEGER,
                computed REAL,
                expected REAL
            );
            CREATE TABLE operations (
                op_id INTEGER PRIMARY KEY,
                input_hash INTEGER,
                output_hash INTEGER,
                silicon TEXT,
                dimension INTEGER,
                latency_ms REAL,
                bandwidth_gbps REAL,
                verified INTEGER
            );
            CREATE TABLE summary (
                metric TEXT PRIMARY KEY,
                value REAL
            );
        """)
        self.conn.commit()

    def record_constants(self) -> None:
        """Record golden hierarchy constants at cycle start."""
        assert self.conn is not None
        omega_sq = OMEGA ** 2  # 1/PHI^2
        constants = [
            ("PHI", PHI, "Golden ratio"),
            ("OMEGA", OMEGA, "1/PHI"),
            ("OMEGA_SQ", omega_sq, "1/PHI^2"),
            ("BETA", BETA, "1/PHI^3"),
            ("GAMMA", GAMMA, "1/PHI^4"),
            ("LUCAS_TOTAL", float(TOTAL_STATES), "Sum L(1..12)"),
            ("BRAHIM_CENTER", float(BRAHIM_CENTER), "C=107"),
            ("BRAHIM_SUM", float(BRAHIM_SUM), "S=214"),
            ("PHI_PI_GAP", PHI_PI_GAP, "Creativity margin"),
            ("CLOSURE", CLOSURE_EXPONENT, "2/3 productive"),
        ]
        self.conn.executemany(
            "INSERT INTO constants VALUES (?,?,?)", constants
        )
        self.conn.commit()

    def verify_axioms(self) -> dict[str, bool]:
        """Verify Universal Structure axioms as pre-conditions."""
        assert self.conn is not None
        results: dict[str, bool] = {}
        omega_sq = OMEGA ** 2  # 1/PHI^2

        # A1: Self-Similarity (OMEGA^2 / BETA = PHI)
        a1 = abs(omega_sq / BETA - PHI) < 1e-14
        results["A1"] = a1

        # A2: Mirror Symmetry
        seq = (27, 42, 60, 75, 97, 117, 139, 154, 172, 187)
        a2 = all((214 - b) in seq for b in seq)
        results["A2"] = a2

        # A3: Lucas Capacity
        a3 = TOTAL_STATES == 840
        results["A3"] = a3

        # A4: Energy Conservation (E(x) = 2*PI)
        x = 0.5
        d = -math.log(x) / math.log(PHI)
        energy = PHI ** d * 2 * math.pi * x
        a4 = abs(energy - 2 * math.pi) < 1e-10
        results["A4"] = a4

        # A5: Dimensional Closure
        a5 = abs(5 / 12 + 1 / 4 - 2 / 3) < 1e-15
        results["A5"] = a5

        # A6: Critical Balance
        a6 = BRAHIM_CENTER / BRAHIM_SUM == 0.5
        results["A6"] = a6

        axiom_data = [
            ("A1", "Self-Similarity", int(a1),
             omega_sq / BETA, PHI),
            ("A2", "Mirror Symmetry", int(a2), 1.0, 1.0),
            ("A3", "Lucas Capacity", int(a3),
             float(TOTAL_STATES), 840.0),
            ("A4", "Energy Conservation", int(a4),
             energy, 2 * math.pi),
            ("A5", "Dimensional Closure", int(a5),
             5 / 12 + 1 / 4, 2 / 3),
            ("A6", "Critical Balance", int(a6),
             BRAHIM_CENTER / BRAHIM_SUM, 0.5),
        ]
        self.conn.executemany(
            "INSERT INTO axioms VALUES (?,?,?,?,?)", axiom_data
        )
        self.conn.commit()
        return results

    def record_operation(self, op: OperationRecord) -> None:
        """Record a single SPIO operation."""
        assert self.conn is not None
        self.conn.execute(
            "INSERT INTO operations VALUES (?,?,?,?,?,?,?,?)",
            (op.op_id, op.input_hash, op.output_hash,
             op.silicon, op.dimension, op.latency_ms,
             op.bandwidth_gbps, int(op.verified)),
        )
        self._op_count += 1
        if self._op_count % 1000 == 0:
            self.conn.commit()

    def record_operations_batch(
        self, ops: list[OperationRecord],
    ) -> None:
        """Record multiple operations efficiently."""
        assert self.conn is not None
        self.conn.executemany(
            "INSERT INTO operations VALUES (?,?,?,?,?,?,?,?)",
            [
                (op.op_id, op.input_hash, op.output_hash,
                 op.silicon, op.dimension, op.latency_ms,
                 op.bandwidth_gbps, int(op.verified))
                for op in ops
            ],
        )
        self._op_count += len(ops)
        self.conn.commit()

    def finalize(self) -> dict[str, float]:
        """Compute and store summary statistics."""
        assert self.conn is not None
        elapsed = time.time() - self._start_time
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) FROM operations")
        total_ops = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM operations WHERE verified=1"
        )
        verified_ops = cur.fetchone()[0]

        cur.execute(
            "SELECT AVG(latency_ms), MIN(latency_ms), "
            "MAX(latency_ms) FROM operations"
        )
        lat = cur.fetchone()
        avg_lat = lat[0] or 0
        min_lat = lat[1] or 0
        max_lat = lat[2] or 0

        cur.execute(
            "SELECT AVG(bandwidth_gbps) FROM operations"
        )
        avg_bw = cur.fetchone()[0] or 0

        cur.execute(
            "SELECT COUNT(*) FROM axioms WHERE verified=1"
        )
        axioms_ok = cur.fetchone()[0]

        cur.execute(
            "SELECT silicon, COUNT(*) FROM operations "
            "GROUP BY silicon"
        )
        silicon_dist = {
            row[0]: row[1] for row in cur.fetchall()
        }

        summary: dict[str, float] = {
            "total_operations": total_ops,
            "verified_operations": verified_ops,
            "axioms_verified": axioms_ok,
            "avg_latency_ms": avg_lat,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat,
            "avg_bandwidth_gbps": avg_bw,
            "elapsed_seconds": elapsed,
            "ops_npu": silicon_dist.get("NPU", 0),
            "ops_cpu": silicon_dist.get("CPU", 0),
            "ops_gpu": silicon_dist.get("GPU", 0),
        }

        self.conn.executemany(
            "INSERT INTO summary VALUES (?,?)",
            [(k, v) for k, v in summary.items()],
        )

        self.conn.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                ("schema_version", self.SCHEMA_VERSION),
                ("created",
                 time.strftime("%Y-%m-%d %H:%M:%S")),
                ("framework", "SPIO Brahim Calculator"),
                ("elapsed_s", str(elapsed)),
            ],
        )
        self.conn.commit()
        return summary

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def size_kb(self) -> float:
        """Return database file size in KB."""
        if os.path.exists(self.path):
            return os.path.getsize(self.path) / 1024
        return 0

    def __enter__(self) -> EvidenceDB:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
