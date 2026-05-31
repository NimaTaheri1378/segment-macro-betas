from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from typing import Any


def mask_username(username: str | None) -> str | None:
    if not username:
        return None
    if len(username) <= 3:
        return username[0] + "***"
    return username[:2] + "***" + username[-1]


def pgpass_metadata(path: Path | None = None) -> dict[str, Any]:
    pgpass = path or (Path.home() / ".pgpass")
    meta: dict[str, Any] = {"exists": pgpass.exists(), "permissions": None, "wrds_username_masked": None}
    if not pgpass.exists():
        return meta
    meta["permissions"] = oct(stat.S_IMODE(pgpass.stat().st_mode))[-3:]
    username = wrds_username_from_pgpass(pgpass)
    meta["wrds_username_masked"] = mask_username(username)
    return meta


def wrds_username_from_pgpass(path: Path | None = None) -> str | None:
    pgpass = path or (Path.home() / ".pgpass")
    if not pgpass.exists():
        return None
    for line in pgpass.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) >= 5 and "wrds" in parts[0].lower():
            return parts[3]
    return None


def connect_wrds(statement_timeout: str = "180s"):
    import wrds

    username = wrds_username_from_pgpass()
    db = wrds.Connection(wrds_username=username)
    try:
        db.raw_sql(f"set statement_timeout to '{statement_timeout}'")
    except Exception:
        pass
    return db


def qident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def qliteral(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def table_ref(library: str, table: str) -> str:
    return f"{qident(library)}.{qident(table)}"


def query_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]
