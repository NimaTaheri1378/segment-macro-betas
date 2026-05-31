from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PyYAML is required to read full YAML configs") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded or {}


def load_schema_roles(path: Path) -> dict[str, dict[str, str]]:
    """Read the selected WRDS library/table/date columns from schema_map.yml.

    Uses PyYAML when available. The regex fallback is intentionally narrow and
    only extracts the selected contract needed by smoke/full extraction code.
    """

    try:
        data = load_yaml(path)
        roles = data.get("roles", {})
        result: dict[str, dict[str, str]] = {}
        for role, entry in roles.items():
            selected = entry.get("selected") or {}
            result[role] = {
                "library": str(selected.get("library", "")),
                "table": str(selected.get("table", "")),
                "date_column": str(selected.get("date_column", "")),
            }
        return result
    except RuntimeError:
        pass

    text = path.read_text(encoding="utf-8")
    roles: dict[str, dict[str, str]] = {}
    current_role: str | None = None
    in_selected = False
    for line in text.splitlines():
        role_match = re.match(r"^  ([A-Za-z0-9_]+):\s*$", line)
        if role_match:
            current_role = role_match.group(1)
            roles[current_role] = {}
            in_selected = False
            continue
        if current_role and line.strip() == "selected:":
            in_selected = True
            continue
        if current_role and in_selected:
            match = re.match(r"^      (library|table|date_column):\s*(.+?)\s*$", line)
            if match:
                roles[current_role][match.group(1)] = match.group(2).strip('"')
            elif re.match(r"^    [A-Za-z_]+:", line):
                in_selected = False
    return roles
