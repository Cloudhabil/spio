"""
Art. 15 — Accuracy, robustness, and cybersecurity.

Input validation at the inference boundary and model/config
integrity verification.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog

# Max input length (characters) — prevent resource exhaustion
_MAX_INPUT_LENGTH = 100_000

# Injection patterns (conservative, deliberately broad)
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<script[\s>]", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(
        r"(\b(DROP|DELETE|ALTER|TRUNCATE)\s+(TABLE|DATABASE)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"\x00"),  # null bytes
]


class InputValidator:
    """Validates inputs before they reach the intelligence layer."""

    def check(
        self, text: str, context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns validation result dict.

        Keys: clean (bool), flags (list), blocked (bool), reason (str).
        """
        flags: list[str] = []

        # Length check
        if len(text) > _MAX_INPUT_LENGTH:
            return {
                "clean": False,
                "flags": ["excessive_length"],
                "blocked": True,
                "reason": (
                    f"Input exceeds {_MAX_INPUT_LENGTH} characters"
                ),
            }

        # Empty input
        if not text.strip():
            return {
                "clean": False,
                "flags": ["empty_input"],
                "blocked": True,
                "reason": "Empty input",
            }

        # Encoding check — ensure valid UTF-8
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            flags.append("encoding_error")

        # Injection patterns
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                flags.append(f"injection:{pattern.pattern[:30]}")

        blocked = any(f.startswith("injection:") for f in flags)
        reason = ""
        if blocked:
            reason = "Potential injection pattern detected"
        elif flags:
            reason = f"Input flagged: {', '.join(flags)}"

        return {
            "clean": len(flags) == 0,
            "flags": flags,
            "blocked": blocked,
            "reason": reason,
        }


class ModelIntegrityVerifier:
    """Verifies model and config haven't been tampered with."""

    def __init__(self, audit_log: ComplianceAuditLog) -> None:
        self._audit = audit_log
        self._baselines: dict[str, str] = {}

    def register_baseline(
        self, model_id: str, config_hash: str,
    ) -> None:
        """Register a known-good hash for a model/config."""
        self._baselines[model_id] = config_hash
        self._audit.record(
            event_type="model_baseline_registered",
            actor="system",
            detail={
                "model_id": model_id,
                "config_hash": config_hash,
            },
        )

    def verify(
        self, model_id: str, current_hash: str,
    ) -> dict[str, Any]:
        """Compare current hash to baseline.  Logged."""
        baseline = self._baselines.get(model_id)
        if baseline is None:
            result = {
                "model_id": model_id,
                "verified": False,
                "reason": "no baseline registered",
            }
        elif baseline == current_hash:
            result = {
                "model_id": model_id,
                "verified": True,
                "reason": "hash matches baseline",
            }
        else:
            result = {
                "model_id": model_id,
                "verified": False,
                "reason": "hash mismatch — possible tampering",
                "expected": baseline,
                "actual": current_hash,
            }
        self._audit.record(
            event_type="model_integrity_check",
            actor="system",
            detail=result,
        )
        return result

    @staticmethod
    def hash_config(config_text: str) -> str:
        """Utility: SHA-256 a config string."""
        return hashlib.sha256(config_text.encode()).hexdigest()
