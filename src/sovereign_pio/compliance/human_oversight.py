"""
Art. 14 — Human oversight.

Three components:
    GuardianGate       – deterministic safety gate on every input
    OversightInterface – human override / confirmation flow
    StopSwitch         – emergency halt with state preservation
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sovereign_pio.compliance.constants import (
    GUARDIAN_TIMEOUT_MS,
    SENSITIVITY_CONFIDENTIAL,
    SENSITIVITY_PUBLIC,
    SENSITIVITY_RESTRICTED,
)
from sovereign_pio.compliance.record_keeping import ComplianceAuditLog


@dataclass
class GuardianDecision:
    """Result of a guardian gate evaluation."""

    action: str            # "allow", "block", "escalate"
    reason: str
    sensitivity: int       # 0-3
    confidence: float
    timestamp: float = field(default_factory=time.time)
    actor: str = "guardian"
    decision_id: str = field(
        default_factory=lambda: uuid.uuid4().hex,
    )


# Built-in sensitivity keywords (conservative defaults)
_PII_KEYWORDS = frozenset({
    "ssn", "social security", "passport", "credit card",
    "bank account", "medical record", "diagnosis",
    "date of birth", "national id",
})
_CONFIDENTIAL_KEYWORDS = frozenset({
    "confidential", "internal only", "trade secret",
    "proprietary", "classified",
})


PolicyFn = Callable[[str, dict[str, Any]], GuardianDecision | None]


class GuardianGate:
    """Tier 1: Evaluates every input BEFORE any model sees it."""

    def __init__(self, audit_log: ComplianceAuditLog) -> None:
        self._audit = audit_log
        self._policies: list[tuple[str, PolicyFn]] = []
        self._timeout_ms = GUARDIAN_TIMEOUT_MS

    def register_policy(
        self, name: str, check_fn: PolicyFn,
    ) -> None:
        """Register a policy check.  First block wins."""
        self._policies.append((name, check_fn))

    def evaluate(
        self,
        request: str,
        actor: str,
        context: dict[str, Any],
    ) -> GuardianDecision:
        """Deterministic safety evaluation.  Always logged."""
        start = time.time()

        # Run registered policies first
        for policy_name, fn in self._policies:
            result = fn(request, context)
            if result is not None and result.action != "allow":
                result.actor = f"guardian/policy:{policy_name}"
                self._log(result, actor, request)
                return result

        # Built-in sensitivity classification
        sensitivity = self._classify_sensitivity(request)

        # Default: allow
        decision = GuardianDecision(
            action="allow",
            reason="passed all checks",
            sensitivity=sensitivity,
            confidence=1.0,
            timestamp=start,
        )

        elapsed_ms = (time.time() - start) * 1000
        if elapsed_ms > self._timeout_ms:
            decision = GuardianDecision(
                action="escalate",
                reason=(
                    f"guardian evaluation exceeded "
                    f"{self._timeout_ms}ms timeout"
                ),
                sensitivity=sensitivity,
                confidence=0.5,
                timestamp=start,
            )

        self._log(decision, actor, request)
        return decision

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_sensitivity(text: str) -> int:
        lower = text.lower()
        if any(kw in lower for kw in _PII_KEYWORDS):
            return SENSITIVITY_RESTRICTED
        if any(kw in lower for kw in _CONFIDENTIAL_KEYWORDS):
            return SENSITIVITY_CONFIDENTIAL
        return SENSITIVITY_PUBLIC

    def _log(
        self,
        decision: GuardianDecision,
        actor: str,
        request: str,
    ) -> None:
        input_hash = hashlib.sha256(request.encode()).hexdigest()
        self._audit.record(
            event_type="guardian_decision",
            actor=actor,
            detail={
                "action": decision.action,
                "reason": decision.reason,
                "sensitivity": decision.sensitivity,
                "confidence": decision.confidence,
                "decision_id": decision.decision_id,
            },
            input_hash=input_hash,
            decision=decision.action,
        )


class OversightInterface:
    """Human oversight capabilities required by Art. 14."""

    def __init__(self, audit_log: ComplianceAuditLog) -> None:
        self._audit = audit_log
        self._pending: dict[str, dict[str, Any]] = {}

    def override(
        self,
        decision_id: str,
        human_id: str,
        new_decision: str,
        reason: str,
    ) -> dict[str, Any]:
        """Human overrides a system decision.  Logged."""
        record = {
            "decision_id": decision_id,
            "human_id": human_id,
            "new_decision": new_decision,
            "reason": reason,
            "timestamp": time.time(),
        }
        self._audit.record(
            event_type="human_override",
            actor=human_id,
            detail=record,
            decision=new_decision,
            oversight_action="override",
        )
        return record

    def require_confirmation(
        self,
        action_desc: str,
        risk_level: int,
    ) -> str:
        """Create a pending confirmation.  Returns confirmation_id."""
        cid = uuid.uuid4().hex
        self._pending[cid] = {
            "action": action_desc,
            "risk_level": risk_level,
            "created": time.time(),
            "status": "pending",
        }
        self._audit.record(
            event_type="confirmation_requested",
            actor="system",
            detail={
                "confirmation_id": cid,
                "action": action_desc,
                "risk_level": risk_level,
            },
        )
        return cid

    def confirm(
        self, confirmation_id: str, human_id: str,
    ) -> dict[str, Any]:
        """Confirm a pending action."""
        entry = self._pending.get(confirmation_id)
        if entry is None:
            return {"error": "unknown confirmation_id"}
        entry["status"] = "confirmed"
        entry["confirmed_by"] = human_id
        entry["confirmed_at"] = time.time()
        self._audit.record(
            event_type="confirmation_granted",
            actor=human_id,
            detail={"confirmation_id": confirmation_id},
            oversight_action="confirm",
        )
        return entry

    def list_pending(self) -> list[dict[str, Any]]:
        """Pending actions awaiting human confirmation."""
        return [
            {"id": cid, **data}
            for cid, data in self._pending.items()
            if data["status"] == "pending"
        ]


class StopSwitch:
    """Emergency halt.  Art. 14 requires this capability."""

    def __init__(self, audit_log: ComplianceAuditLog) -> None:
        self._audit = audit_log
        self._halted = False
        self._halt_record: dict[str, Any] | None = None

    def halt(self, reason: str, actor: str) -> dict[str, Any]:
        """Immediate stop.  Preserves state for incident investigation."""
        self._halted = True
        self._halt_record = {
            "reason": reason,
            "actor": actor,
            "timestamp": time.time(),
        }
        self._audit.record(
            event_type="emergency_halt",
            actor=actor,
            detail=self._halt_record,
            decision="halt",
            oversight_action="halt",
        )
        return self._halt_record

    def is_halted(self) -> bool:
        return self._halted

    def resume(self, actor: str, reason: str) -> dict[str, Any]:
        """Resume after halt.  Requires human actor."""
        record = {
            "reason": reason,
            "actor": actor,
            "timestamp": time.time(),
            "previous_halt": self._halt_record,
        }
        self._halted = False
        self._halt_record = None
        self._audit.record(
            event_type="halt_resumed",
            actor=actor,
            detail=record,
            decision="resume",
            oversight_action="resume",
        )
        return record
