"""
N-Body Scale Hierarchy - Resource quantizer from Brahim subset sums.

Provides 369-level quantization derived from the 10 Brahim numbers.
Used by IIAS and DAG planner to enforce the N<=27 agent ceiling
and assign deterministic Brahim scales to agent pairs.

Reference: DOI 10.5281/zenodo.18437705
"""

from __future__ import annotations

import bisect
from itertools import combinations

from sovereign_pio.constants import (
    BRAHIM_CENTER,
    BRAHIM_NUMBERS,
    MAX_CONCURRENT_AGENTS,
    MIRROR_CONSTANT,
    PHI,
    PRODUCT_INVARIANT_EXPONENT,
    TRIANGLE_SILICON,
)


class ScaleHierarchy:
    """369-level resource quantizer from Brahim subset sums."""

    def __init__(self) -> None:
        self._levels: list[set[int]] = []
        all_sums: set[int] = set()
        for k in range(1, 11):
            level_sums = {
                sum(c) for c in combinations(BRAHIM_NUMBERS, k)
            }
            self._levels.append(level_sums)
            all_sums |= level_sums
        self._scales: list[int] = sorted(all_sums)

    @property
    def scales(self) -> list[int]:
        """Sorted list of 369 distinct Brahim scales."""
        return list(self._scales)

    def quantize(self, value: float, total: float) -> int:
        """Snap a resource fraction to nearest Brahim scale."""
        if total <= 0:
            return self._scales[0]
        target = int(round(value / total * self._scales[-1]))
        idx = bisect.bisect_left(self._scales, target)
        if idx == 0:
            return self._scales[0]
        if idx >= len(self._scales):
            return self._scales[-1]
        lo = self._scales[idx - 1]
        hi = self._scales[idx]
        return lo if (target - lo) <= (hi - target) else hi

    def level_for_n(self, n: int) -> int | None:
        """Minimum level needed to support N agents."""
        pairs = n * (n - 1) // 2
        cumulative = 0
        for k, level_sums in enumerate(self._levels, 1):
            cumulative += len(level_sums)
            if cumulative >= pairs:
                return k
        return None

    def validate_concurrency(self, n: int) -> dict:
        """Check N <= 27, return proof data."""
        pairs = n * (n - 1) // 2
        ceiling = MAX_CONCURRENT_AGENTS
        total_scales = len(self._scales)
        valid = n <= ceiling and pairs <= total_scales
        return {
            "n": n,
            "pairs": pairs,
            "ceiling": ceiling,
            "scales_available": total_scales,
            "valid": valid,
            "level_needed": self.level_for_n(n),
        }

    def silicon_triangle(self) -> dict:
        """Return {NPU: 42, CPU: 75, GPU: 97} with conservation."""
        tri = dict(TRIANGLE_SILICON)
        tri["sum"] = sum(TRIANGLE_SILICON.values())
        tri["equals_mirror"] = tri["sum"] == MIRROR_CONSTANT
        return tri

    def audit_product(self, scale_list: list[int]) -> dict:
        """Verify phi^(-sum(scales)) matches expected invariant."""
        exponent = sum(scale_list)
        actual = PHI ** (-exponent)
        expected = PHI ** (-PRODUCT_INVARIANT_EXPONENT)
        match = exponent == PRODUCT_INVARIANT_EXPONENT
        return {
            "exponent": exponent,
            "expected_exponent": PRODUCT_INVARIANT_EXPONENT,
            "value": actual,
            "expected_value": expected,
            "match": match,
        }

    def modular_coverage(self) -> dict:
        """93/107 residue coverage mod BRAHIM_CENTER."""
        residues = {s % BRAHIM_CENTER for s in self._scales}
        return {
            "modulus": BRAHIM_CENTER,
            "covered": len(residues),
            "total": BRAHIM_CENTER,
            "residues": sorted(residues),
        }
