"""P4-0 Evidence Board store (Redis-backed snapshot, fail-open).

In-memory dict is authoritative for the process lifetime; Redis holds a
best-effort snapshot (meta without board lists, plus evidence/claims lists)
so a restart can reassemble the investigation. Storage errors never raise.
"""

import json
from typing import Any

from pydantic import TypeAdapter

from app.evidence.models import Board, Claim, Evidence, Investigation
from app.rediskit import holder
from app.utils.logger import get_logger

logger = get_logger(__name__)

_evidence_list_adapter: TypeAdapter[list[Evidence]] = TypeAdapter(list[Evidence])
_claims_list_adapter: TypeAdapter[list[Claim]] = TypeAdapter(list[Claim])


def _meta_key(investigation_id: str) -> str:
    return f"argus:inv:{investigation_id}:meta"


def _evidence_key(investigation_id: str) -> str:
    return f"argus:inv:{investigation_id}:evidence"


def _claims_key(investigation_id: str) -> str:
    return f"argus:inv:{investigation_id}:claims"


def _as_str(raw: str | bytes) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


class EvidenceBoardStore:
    def __init__(self) -> None:
        self._memory: dict[str, Investigation] = {}

    async def save(self, inv: Investigation, ttl_s: int) -> None:
        """Mirror in memory, then best-effort Redis snapshot. Never raises."""
        self._memory[inv.id] = inv
        client = holder.client
        if client is None:
            return
        try:
            meta: dict[str, Any] = inv.model_dump(mode="json", exclude={"board"})
            meta["evidence_count"] = len(inv.board.evidence)
            meta["claim_count"] = len(inv.board.claims)
            meta_json: str = json.dumps(meta)
            evidence_json: str = _evidence_list_adapter.dump_json(inv.board.evidence).decode("utf-8")
            claims_json: str = _claims_list_adapter.dump_json(inv.board.claims).decode("utf-8")
            await client.set(_meta_key(inv.id), meta_json)
            await client.set(_evidence_key(inv.id), evidence_json)
            await client.set(_claims_key(inv.id), claims_json)
            await client.expire(_meta_key(inv.id), ttl_s)
            await client.expire(_evidence_key(inv.id), ttl_s)
            await client.expire(_claims_key(inv.id), ttl_s)
        except Exception as exc:  # noqa: BLE001 - fail open, always
            logger.warning({"message": "EvidenceBoard save failed (ignored)", "error": str(exc)})

    async def load(self, investigation_id: str) -> Investigation | None:
        """Memory hit returns immediately; else reassemble from Redis snapshot."""
        cached: Investigation | None = self._memory.get(investigation_id)
        if cached is not None:
            return cached
        client = holder.client
        if client is None:
            return None
        try:
            raw_meta = await client.get(_meta_key(investigation_id))
            if not raw_meta:
                return None
            meta: dict[str, Any] = json.loads(_as_str(raw_meta))
            meta.pop("evidence_count", None)
            meta.pop("claim_count", None)
            evidence: list[Evidence] = []
            claims: list[Claim] = []
            raw_ev = await client.get(_evidence_key(investigation_id))
            if raw_ev:
                evidence = _evidence_list_adapter.validate_json(_as_str(raw_ev))
            raw_cl = await client.get(_claims_key(investigation_id))
            if raw_cl:
                claims = _claims_list_adapter.validate_json(_as_str(raw_cl))
            meta["board"] = Board(evidence=evidence, claims=claims).model_dump(mode="json")
            return Investigation.model_validate(meta)
        except Exception as exc:  # noqa: BLE001 - fail open, always
            logger.warning({"message": "EvidenceBoard load failed (ignored)", "error": str(exc)})
            return None

    async def list_recent(self, limit: int) -> list[Investigation]:
        """Newest-first snapshot list. Never raises."""
        try:
            clamped = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            clamped = 20
        try:
            client = holder.client
            if client is None:
                items = sorted(
                    self._memory.values(), key=lambda inv: inv.created_at, reverse=True
                )
                return items[:clamped]
            scanned_keys: list[str] = []
            cursor: int = 0
            while True:
                try:
                    cursor, keys = await client.scan(
                        cursor=cursor, match="argus:inv:*:meta", count=100
                    )
                except Exception as exc:  # noqa: BLE001 - fail open, always
                    logger.warning(
                        {"message": "EvidenceBoard list scan failed (ignored)", "error": str(exc)}
                    )
                    break
                for key in keys or []:
                    scanned_keys.append(_as_str(key))
                    if len(scanned_keys) >= 500:
                        break
                if len(scanned_keys) >= 500:
                    break
                try:
                    cursor = int(cursor)
                except (TypeError, ValueError):
                    break
                if cursor == 0:
                    break
            prefix, suffix = "argus:inv:", ":meta"
            ids: set[str] = set(self._memory.keys())
            for key in scanned_keys:
                if key.startswith(prefix) and key.endswith(suffix):
                    ids.add(key[len(prefix) : -len(suffix)])
            found: list[Investigation] = []
            for investigation_id in ids:
                try:
                    inv = await self.load(investigation_id)
                except Exception:  # noqa: BLE001 - drop failures silently
                    continue
                if inv is not None:
                    found.append(inv)
            found.sort(key=lambda inv: inv.created_at, reverse=True)
            return found[:clamped]
        except Exception as exc:  # noqa: BLE001 - fail open, always
            logger.warning({"message": "EvidenceBoard list failed (ignored)", "error": str(exc)})
            try:
                fallback = sorted(
                    self._memory.values(), key=lambda inv: inv.created_at, reverse=True
                )
                return fallback[:clamped]
            except Exception:  # noqa: BLE001 - never raises
                return []
