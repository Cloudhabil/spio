"""
VNAND Audit Trail Generator.

Generates cryptographic hashes (VNAND resonance fingerprints) for
Brahim's Laws computations, ensuring reproducibility and verification.

The VNAND pattern creates immutable fingerprints encoding:
- Input parameters
- Computation states
- Results and timestamps
- Full audit trails

Author: Elias Oulad Brahim
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from ..models.curve_data import EllipticCurveData
from ..models.analysis_result import BrahimAnalysisResult

logger = logging.getLogger(__name__)


class VNANDHasher:
    """
    Generates VNAND resonance hashes for computation verification.

    VNAND (Virtual NAND) hashing creates immutable fingerprints that encode:
    - Complete computation state
    - Input/output relationships
    - Temporal ordering
    - Audit trail for reproducibility

    Example:
        hasher = VNANDHasher()
        result_hash = hasher.hash_analysis(result)
        batch_hash = hasher.hash_batch(results)
        hasher.save_audit_trail(output_dir)
    """

    def __init__(self, algorithm: str = 'sha256'):
        """
        Initialize hasher.

        Args:
            algorithm: Hash algorithm ('sha256', 'sha384', 'sha512')
        """
        self.algorithm = algorithm
        self.hash_log: List[Dict[str, Any]] = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==========================================================================
    # CORE HASHING
    # ==========================================================================

    def _compute_hash(self, data: str) -> str:
        """Compute hash of string data."""
        if self.algorithm == 'sha256':
            return hashlib.sha256(data.encode()).hexdigest()
        elif self.algorithm == 'sha384':
            return hashlib.sha384(data.encode()).hexdigest()
        elif self.algorithm == 'sha512':
            return hashlib.sha512(data.encode()).hexdigest()
        else:
            return hashlib.sha256(data.encode()).hexdigest()

    def hash_state(
        self,
        phase_name: str,
        state_dict: Dict[str, Any]
    ) -> str:
        """
        Generate hash of a computation phase state.

        Args:
            phase_name: Name of computation phase
            state_dict: State dictionary to hash

        Returns:
            Hexadecimal hash string
        """
        # Serialize state deterministically
        state_json = json.dumps(state_dict, sort_keys=True, default=str)

        # Combine phase name and state
        h = hashlib.new(self.algorithm)
        h.update(phase_name.encode())
        h.update(b":")
        h.update(state_json.encode())

        phase_hash = h.hexdigest()

        # Log the hash
        self.hash_log.append({
            "phase": phase_name,
            "hash": phase_hash,
            "timestamp": datetime.now().isoformat(),
            "state_keys": list(state_dict.keys()),
            "state_size": len(state_json)
        })

        return phase_hash

    # ==========================================================================
    # CURVE HASHING
    # ==========================================================================

    def hash_curve(self, curve: EllipticCurveData) -> str:
        """
        Generate hash for an elliptic curve.

        Args:
            curve: Elliptic curve data

        Returns:
            Curve fingerprint hash
        """
        state = {
            "label": curve.label,
            "a_invariants": curve.a_invariants,
            "conductor": curve.conductor,
            "rank": curve.rank,
            "torsion_order": curve.torsion_order,
            "tamagawa_product": curve.tamagawa_product,
            "real_period": curve.real_period,
            "im_tau": curve.im_tau,
            "sha_analytic": curve.sha_analytic,
            "source": curve.source
        }

        return self.hash_state(f"curve_{curve.label}", state)

    # ==========================================================================
    # ANALYSIS HASHING
    # ==========================================================================

    def hash_analysis(self, result: BrahimAnalysisResult) -> str:
        """
        Generate hash for a complete analysis result.

        This creates a fingerprint encoding:
        - Input curve
        - All computed law values
        - Errors and consistency

        Args:
            result: Analysis result

        Returns:
            Analysis fingerprint hash
        """
        state = {
            # Curve identification
            "curve_label": result.curve.label,
            "conductor": result.curve.conductor,

            # Law 1
            "sha_median_predicted": result.sha_median_predicted,
            "sha_omega_predicted": result.sha_omega_predicted,
            "law1_error": result.law1_error,

            # Law 2
            "reynolds_number": result.reynolds_number,

            # Law 3
            "regime": result.regime.value,

            # Law 4
            "sha_max_predicted": result.sha_max_predicted,
            "law4_error": result.law4_error,

            # Law 5
            "p_scaling_exponent": result.p_scaling_exponent,

            # Law 6
            "is_consistent": result.is_consistent,

            # Timestamp
            "timestamp": result.timestamp
        }

        analysis_hash = self.hash_state(
            f"analysis_{result.curve.label}",
            state
        )

        return analysis_hash

    def hash_and_attach(self, result: BrahimAnalysisResult) -> BrahimAnalysisResult:
        """
        Generate hash and attach to result.

        Args:
            result: Analysis result (modified in place)

        Returns:
            Same result with vnand_hash populated
        """
        result.vnand_hash = self.hash_analysis(result)
        return result

    # ==========================================================================
    # BATCH HASHING
    # ==========================================================================

    def hash_batch(
        self,
        results: List[BrahimAnalysisResult],
        attach: bool = True
    ) -> Dict[str, Any]:
        """
        Generate hashes for batch of results.

        Args:
            results: List of analysis results
            attach: Whether to attach hashes to results

        Returns:
            Batch audit information including master hash
        """
        individual_hashes = []

        for result in results:
            h = self.hash_analysis(result)
            if attach:
                result.vnand_hash = h
            individual_hashes.append(h)

        # Compute batch master hash
        batch_state = {
            "total_curves": len(results),
            "individual_hashes": individual_hashes[:100],  # First 100 for state
            "hash_count": len(individual_hashes),
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id
        }

        master_hash = self.hash_state("batch_master", batch_state)

        return {
            "master_hash": master_hash,
            "total_curves": len(results),
            "individual_hashes": individual_hashes,
            "timestamp": datetime.now().isoformat(),
            "session_id": self._session_id
        }

    # ==========================================================================
    # AUDIT TRAIL
    # ==========================================================================

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return complete audit trail."""
        return self.hash_log.copy()

    def save_audit_trail(
        self,
        output_dir: Path,
        filename: Optional[str] = None
    ) -> Path:
        """
        Save audit trail to disk.

        Args:
            output_dir: Output directory
            filename: Custom filename (auto-generated if None)

        Returns:
            Path to saved audit file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"vnand_audit_trail_{self._session_id}.json"

        audit_file = output_dir / filename

        audit_data = {
            "session_id": self._session_id,
            "algorithm": self.algorithm,
            "total_hashes": len(self.hash_log),
            "created_at": datetime.now().isoformat(),
            "hash_log": self.hash_log
        }

        with open(audit_file, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2)

        logger.info(f"Audit trail saved to {audit_file}")
        return audit_file

    def load_audit_trail(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Load audit trail from file.

        Args:
            filepath: Path to audit file

        Returns:
            Loaded hash log
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data.get('hash_log', [])

    # ==========================================================================
    # VERIFICATION
    # ==========================================================================

    def verify_result(
        self,
        result: BrahimAnalysisResult
    ) -> Tuple[bool, str]:
        """
        Verify that a result's hash matches recomputation.

        Args:
            result: Result to verify

        Returns:
            (is_valid, message)
        """
        if not result.vnand_hash:
            return (False, "No hash attached to result")

        # Recompute hash
        recomputed = self.hash_analysis(result)

        if recomputed == result.vnand_hash:
            return (True, "Hash verification passed")
        else:
            return (False, f"Hash mismatch: expected {result.vnand_hash[:16]}..., got {recomputed[:16]}...")

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def clear(self):
        """Clear hash log."""
        self.hash_log.clear()

    def __repr__(self) -> str:
        return (
            f"VNANDHasher(algorithm='{self.algorithm}', "
            f"logged_hashes={len(self.hash_log)})"
        )
