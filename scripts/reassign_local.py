"""Reassign legacy "local" investigations to a real subject (P7-1).

Default advice is to NOT run this. Legacy rows age out by TTL on their
own, and the board schema never migrates. This exists only for the case
where an operator must keep pre-scoping rows visible to one user.

Usage:
    venv/Scripts/python.exe scripts/reassign_local.py --to alice --dry-run
    venv/Scripts/python.exe scripts/reassign_local.py --to alice --apply

--dry-run lists what would change and writes nothing. --apply performs
the reassignment. Exactly one of the two is required.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import settings
from app.investigations import InvestigationManager
from app.rediskit import close_redis, connect_redis, holder


async def _run(to_subject: str, apply: bool, limit: int) -> int:
    holder.client = await connect_redis()
    try:
        manager = InvestigationManager()
        rows = await manager._store.list_recent(500)
        local_rows = [row for row in rows if row.user_id == "local"][:limit]
        for row in local_rows:
            print(f"{row.id}  {row.status.value}  {row.query[:80]!r}")
        print(f"{len(local_rows)} local row(s) would move to {to_subject!r}")
        if not apply:
            print("dry-run: nothing written (pass --apply to write)")
            return 0
        for row in local_rows:
            row.user_id = to_subject
            await manager._store.save(row, settings.investigation_ttl_s)
        print(f"reassigned {len(local_rows)} row(s)")
        return 0
    finally:
        await close_redis()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--to", required=True, help="Subject receiving the rows")
    parser.add_argument("--limit", type=int, default=500)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if not args.to.strip():
        print("--to must be non-empty", file=sys.stderr)
        return 2
    return asyncio.run(_run(args.to.strip(), args.apply, max(args.limit, 1)))


if __name__ == "__main__":
    raise SystemExit(main())
