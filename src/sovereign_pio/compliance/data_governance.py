"""
Art. 10 — Data governance + Brahim Industry Label (BIL).

Canonical Python implementation of BUIM's BIL system.
BIL tags every data element with provenance: sector, type, source,
and a check digit.  BUIM APK becomes a consumer via API.

DataGovernance wraps BIL tagging with audit-logged provenance
tracking, sensitivity classification, and source-upgrade workflows.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from sovereign_pio.compliance.record_keeping import ComplianceAuditLog

# ---------------------------------------------------------------------------
# BIL sector codes  (Brahim Sequence)
# ---------------------------------------------------------------------------
BIL_SECTORS: dict[int, str] = {
    27: "Electrical",
    42: "Mechanical",
    60: "Chemical",
    75: "Civil",
    97: "Software",
    117: "Aerospace",
    139: "Biomedical",
    154: "Telecommunications",
    172: "Energy",
    187: "Environmental",
}

# ---------------------------------------------------------------------------
# BIL data-type codes
# ---------------------------------------------------------------------------
BIL_DATA_TYPES: dict[int, str] = {
    1: "Specification",
    2: "Measurement",
    3: "Simulation",
    4: "Certification",
    5: "Test Result",
    6: "Design Document",
    7: "Operational Log",
    8: "Audit Record",
    9: "Learned",
}

# ---------------------------------------------------------------------------
# BIL source codes
# ---------------------------------------------------------------------------
BIL_SOURCES: dict[int, str] = {
    100: "IEC Standard",
    200: "Manufacturer",
    300: "Accredited Lab",
    400: "Regulatory Body",
    500: "Peer-Reviewed",
    600: "Industry Consortium",
    700: "Internal Validated",
    800: "Field Observation",
    900: "ML Prediction",
    999: "Unverified",
}


# ---------------------------------------------------------------------------
# BIL dataclass
# ---------------------------------------------------------------------------
@dataclass
class BrahimIndustryLabel:
    """BIL:<sector>:<type>:<source>:<id>-<check>

    Encodes provenance of any data element used by the AI system.
    """

    sector: int       # 27, 42, … (Brahim sequence)
    data_type: int    # 1–9
    source: int       # 100, 200, … 999
    item_id: str
    check_digit: str

    @property
    def is_deterministic(self) -> bool:
        """Sources below 900 are human-verified / deterministic."""
        return self.source < 900

    def encode(self) -> str:
        """Encode to canonical string ``BIL:27:1:100:60617-3``."""
        return (
            f"BIL:{self.sector}:{self.data_type}:{self.source}"
            f":{self.item_id}-{self.check_digit}"
        )

    @classmethod
    def decode(cls, bil_string: str) -> BrahimIndustryLabel:
        """Parse a ``BIL:…`` string back to object.

        Raises ``ValueError`` on malformed input.
        """
        if not bil_string.startswith("BIL:"):
            raise ValueError(f"Not a BIL string: {bil_string!r}")
        parts = bil_string[4:].split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Expected 4 colon-separated fields after 'BIL:', "
                f"got {len(parts)}"
            )
        sector = int(parts[0])
        data_type = int(parts[1])
        source = int(parts[2])
        id_check = parts[3]
        if "-" not in id_check:
            raise ValueError(
                f"Expected '<item_id>-<check>' in last field, "
                f"got {id_check!r}"
            )
        item_id, check_digit = id_check.rsplit("-", 1)
        return cls(
            sector=sector,
            data_type=data_type,
            source=source,
            item_id=item_id,
            check_digit=check_digit,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "sector": self.sector,
            "sector_name": BIL_SECTORS.get(self.sector, "Unknown"),
            "data_type": self.data_type,
            "data_type_name": BIL_DATA_TYPES.get(self.data_type, "Unknown"),
            "source": self.source,
            "source_name": BIL_SOURCES.get(self.source, "Unknown"),
            "item_id": self.item_id,
            "check_digit": self.check_digit,
            "is_deterministic": self.is_deterministic,
            "encoded": self.encode(),
        }


# ---------------------------------------------------------------------------
# DataGovernance (Art. 10)
# ---------------------------------------------------------------------------
class DataGovernance:
    """Art. 10 — data governance with BIL provenance tracking."""

    def __init__(self, audit_log: ComplianceAuditLog) -> None:
        self._audit = audit_log
        self._tags: dict[str, dict[str, Any]] = {}

    def tag(
        self,
        data: str,
        bil: BrahimIndustryLabel,
        sensitivity: int,
    ) -> str:
        """Tag input data with BIL provenance + sensitivity.

        Returns ``tag_id``.
        """
        tag_id = uuid.uuid4().hex[:12]
        entry: dict[str, Any] = {
            "tag_id": tag_id,
            "bil": bil.encode(),
            "bil_detail": bil.to_dict(),
            "sensitivity": sensitivity,
            "tagged_at": time.time(),
            "data_preview": data[:80] if data else "",
            "valid": True,
        }
        self._tags[tag_id] = entry
        self._audit.record(
            event_type="data_tagged",
            actor="system",
            detail={
                "tag_id": tag_id,
                "bil": bil.encode(),
                "sensitivity": sensitivity,
                "is_deterministic": bil.is_deterministic,
            },
        )
        return tag_id

    def verify_provenance(self, tag_id: str) -> dict[str, Any]:
        """Check if tagged data's source is still valid."""
        entry = self._tags.get(tag_id)
        if entry is None:
            return {"error": "unknown tag_id", "valid": False}
        bil = BrahimIndustryLabel.decode(entry["bil"])
        result: dict[str, Any] = {
            "tag_id": tag_id,
            "bil": entry["bil"],
            "source": bil.source,
            "source_name": BIL_SOURCES.get(bil.source, "Unknown"),
            "is_deterministic": bil.is_deterministic,
            "valid": entry.get("valid", True),
            "checked_at": time.time(),
        }
        self._audit.record(
            event_type="provenance_verified",
            actor="system",
            detail=result,
        )
        return result

    def upgrade_source(
        self,
        tag_id: str,
        new_source: int,
        human_id: str,
    ) -> dict[str, Any]:
        """Human verifies ML prediction -> upgrade BIL source.

        Typical flow: 900 (ML) -> 700 (Internal Validated) or 100 (IEC).
        """
        entry = self._tags.get(tag_id)
        if entry is None:
            return {"error": "unknown tag_id"}
        old_bil = BrahimIndustryLabel.decode(entry["bil"])
        old_source = old_bil.source
        new_bil = BrahimIndustryLabel(
            sector=old_bil.sector,
            data_type=old_bil.data_type,
            source=new_source,
            item_id=old_bil.item_id,
            check_digit=old_bil.check_digit,
        )
        entry["bil"] = new_bil.encode()
        entry["bil_detail"] = new_bil.to_dict()
        entry["upgraded_at"] = time.time()
        entry["upgraded_by"] = human_id
        result: dict[str, Any] = {
            "tag_id": tag_id,
            "old_source": old_source,
            "new_source": new_source,
            "new_bil": new_bil.encode(),
            "is_deterministic": new_bil.is_deterministic,
            "upgraded_by": human_id,
        }
        self._audit.record(
            event_type="source_upgraded",
            actor=human_id,
            detail=result,
            oversight_action="upgrade",
        )
        return result

    def stats(self) -> dict[str, Any]:
        """Counts by source type (deterministic vs ML vs unverified)."""
        deterministic = 0
        ml = 0
        unverified = 0
        for entry in self._tags.values():
            bil = BrahimIndustryLabel.decode(entry["bil"])
            if bil.source < 900:
                deterministic += 1
            elif bil.source == 900:
                ml += 1
            else:
                unverified += 1
        return {
            "total_tagged": len(self._tags),
            "deterministic": deterministic,
            "ml_predicted": ml,
            "unverified": unverified,
            "deterministic_ratio": (
                deterministic / len(self._tags)
                if self._tags
                else 1.0
            ),
        }
