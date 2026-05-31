from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configured_expected_root() -> Path | None:
    raw = os.environ.get("SMB_EXPECTED_PROJECT_ROOT") or os.environ.get("SMB_PROJECT_ROOT")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


ROLE_SPECS: dict[str, dict[str, Any]] = {
    "historical_segments_geo": {
        "description": "Compustat Historical Segments geographic labels",
        "library_terms": ("comp", "seg"),
        "table_terms": ("seg", "segment", "geo", "geog"),
        "preferred_tables": ("wrds_seg_geo", "seg_geo"),
        "required_any": (("gvkey",), ("sid",), ("srcdate", "datadate"), ("gareag", "gareat", "geotp")),
        "date_priority": ("srcdate", "datadate"),
    },
    "historical_segments_values": {
        "description": "Compustat Historical Segments segment values and sales fields",
        "library_terms": ("comp", "seg"),
        "table_terms": ("seg", "segment", "merged", "annfund"),
        "preferred_tables": ("wrds_segmerged", "seg_annfund"),
        "required_any": (("gvkey",), ("sid",), ("srcdate", "datadate"), ("sales", "revts", "ias")),
        "date_priority": ("datadate", "srcdate", "filedate", "rdq"),
    },
    "crsp_monthly_stock": {
        "description": "CRSP monthly stock returns and implementation variables",
        "library_terms": ("crsp",),
        "table_terms": ("msf", "monthly", "stock"),
        "preferred_tables": ("msf", "msf_v2", "monthly_stock_file", "stk_mth"),
        "required_any": (("permno",), ("date", "mthcaldt"), ("ret", "mthret"), ("prc", "mthprc")),
        "date_priority": ("date", "mthcaldt"),
    },
    "crsp_daily_stock": {
        "description": "CRSP daily stock returns for event windows",
        "library_terms": ("crsp",),
        "table_terms": ("dsf", "daily", "stock"),
        "preferred_tables": ("dsf", "dsf_v2", "daily_stock_file", "stk_dly"),
        "required_any": (("permno",), ("date", "dlycaldt"), ("ret", "dlyret"), ("prc", "dlyprc")),
        "date_priority": ("date", "dlycaldt"),
    },
    "ccm_link_history": {
        "description": "CRSP/Compustat Merged link history",
        "library_terms": ("crsp", "ccm", "comp"),
        "table_terms": ("ccm", "link"),
        "preferred_tables": ("ccmxpf_linktable", "ccmxpf_lnkhist", "ccm_lookup"),
        "required_any": (("gvkey",), ("lpermno", "permno"), ("linkdt",), ("linkenddt",)),
        "date_priority": ("linkdt", "linkenddt"),
    },
    "compustat_fundamentals_annual": {
        "description": "Compustat North America annual fundamentals",
        "library_terms": ("comp",),
        "table_terms": ("funda", "fundamentals", "annual"),
        "preferred_tables": ("funda", "fundq"),
        "required_any": (("gvkey",), ("datadate",), ("at", "sale", "revt")),
        "date_priority": ("datadate", "fyear"),
    },
    "benchmark_factors_wrds": {
        "description": "WRDS-hosted factor or Fama/French support tables, if visible",
        "library_terms": ("ff", "fama", "factor"),
        "table_terms": ("factor", "fama", "ff", "momentum"),
        "preferred_tables": ("factors_monthly", "ff_factors_monthly", "ff3_monthly", "ff5_monthly"),
        "required_any": (("date", "caldt", "yyyymm"),),
        "date_priority": ("date", "caldt", "yyyymm"),
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def mask_username(username: str | None) -> str | None:
    if not username:
        return None
    if len(username) <= 3:
        return username[0] + "***"
    return username[:2] + "***" + username[-1]


def safe_under_root(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to write outside project root: {resolved}") from exc
    return resolved


def read_pgpass_metadata() -> dict[str, Any]:
    pgpass = Path.home() / ".pgpass"
    meta: dict[str, Any] = {
        "path": str(pgpass),
        "exists": pgpass.exists(),
        "permissions": None,
        "wrds_username_masked": None,
    }
    if not pgpass.exists():
        return meta

    mode = stat.S_IMODE(pgpass.stat().st_mode)
    meta["permissions"] = oct(mode)[-3:]
    try:
        lines = pgpass.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return meta
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) >= 5 and "wrds" in parts[0].lower():
            meta["wrds_username_masked"] = mask_username(parts[3])
            meta["_wrds_username"] = parts[3]
            break
    return meta


def normalize_columns(columns_df: Any) -> list[dict[str, Any]]:
    records = columns_df.to_dict(orient="records")
    clean: list[dict[str, Any]] = []
    for row in records:
        lowered = {str(k).lower(): v for k, v in row.items()}
        name = lowered.get("name") or lowered.get("column_name") or lowered.get("variable")
        dtype = lowered.get("type") or lowered.get("data_type") or lowered.get("format")
        label = lowered.get("label") or lowered.get("description")
        if name is None:
            continue
        clean.append({"name": str(name).lower(), "type": None if dtype is None else str(dtype), "label": None if label is None else str(label)})
    return clean


def qident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def table_name_score(table: str, spec: dict[str, Any]) -> int:
    t = table.lower()
    score = 0
    for idx, preferred in enumerate(spec["preferred_tables"]):
        if t == preferred:
            score += 100 - idx
        elif preferred in t:
            score += 45 - idx
    for term in spec["table_terms"]:
        if term in t:
            score += 10
    return score


def library_score(library: str, spec: dict[str, Any]) -> int:
    lib = library.lower()
    return sum(8 for term in spec["library_terms"] if term in lib)


def column_score(columns: list[dict[str, Any]], spec: dict[str, Any]) -> tuple[int, list[str], list[list[str]]]:
    names = {c["name"] for c in columns}
    hits: list[str] = []
    missing_groups: list[list[str]] = []
    score = 0
    for group in spec["required_any"]:
        found = [name for name in group if name in names]
        if found:
            hits.extend(found)
            score += 30
        else:
            missing_groups.append(list(group))
    return score, sorted(set(hits)), missing_groups


def pick_date_column(columns: list[dict[str, Any]], spec: dict[str, Any]) -> str | None:
    names = {c["name"] for c in columns}
    for name in spec["date_priority"]:
        if name in names:
            return name
    for name in ("date", "datadate", "caldt", "fyear", "yyyymm"):
        if name in names:
            return name
    return None


def to_yaml(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{pad}[]"
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.append(to_yaml(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{pad}- {json.dumps(item)}")
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{yaml_scalar(value)}"


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or re.search(r"[:#\n\[\]{}&,*!|>'\"%@`]", text) or text.strip() != text:
        return json.dumps(text)
    return text


def markdown_report(manifest: dict[str, Any], schema_map: dict[str, Any]) -> str:
    lines = [
        "# WRDS Schema Audit Report",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Created UTC: `{manifest['created_utc']}`",
        f"- Project root: `{manifest['project_root']}`",
        f"- WRDS connection: `{manifest['wrds']['connection_status']}`",
        f"- Visible WRDS libraries: `{manifest['wrds'].get('library_count')}`",
        "",
        "## Selected Sources",
        "",
        "| Role | Selected table | Date column | Date range | Required hits | Risks |",
        "|---|---|---|---|---|---|",
    ]
    for role, entry in schema_map["roles"].items():
        selected = entry.get("selected") or {}
        if selected:
            table = f"{selected.get('library')}.{selected.get('table')}"
            date_col = selected.get("date_column") or ""
            date_range = selected.get("date_range") or {}
            date_txt = ""
            if date_range:
                date_txt = f"{date_range.get('min')} to {date_range.get('max')}"
            hits = ", ".join(selected.get("required_hits") or [])
            risks = "; ".join(entry.get("risks") or [])
        else:
            table = "UNRESOLVED"
            date_col = ""
            date_txt = ""
            hits = ""
            risks = "; ".join(entry.get("risks") or ["no candidate selected"])
        lines.append(f"| `{role}` | `{table}` | `{date_col}` | {date_txt} | {hits} | {risks} |")

    lines.extend([
        "",
        "## Discipline Notes",
        "",
        "- This run performed schema discovery and limited metadata/date-range probes only.",
        "- No broad table extracts were written.",
        "- Credentials and `.pgpass` contents were not written to the manifest or report.",
        "- Tables marked unresolved or risky should be inspected before any full extraction.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser()
    expected_root = configured_expected_root()
    if expected_root is not None and project_root.resolve() != expected_root:
        raise SystemExit(f"Resolved project root mismatch: {project_root.resolve()} != {expected_root}")
    if Path.cwd().resolve() != project_root.resolve():
        raise SystemExit(f"Refusing to run outside project root: cwd={Path.cwd()}")

    run_root = safe_under_root(project_root, project_root / "runs" / args.run_id)
    manifest_dir = safe_under_root(project_root, run_root / "manifests")
    report_dir = safe_under_root(project_root, run_root / "reports")
    config_dir = safe_under_root(project_root, project_root / "configs")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "created_utc": now_iso(),
        "project_root": str(project_root),
        "python": sys.version.split()[0],
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
        },
        "pgpass": {},
        "wrds": {"connection_status": "not_started"},
        "candidate_libraries": {},
        "roles": {},
        "errors": [],
    }

    pgpass_meta = read_pgpass_metadata()
    username = pgpass_meta.pop("_wrds_username", None)
    manifest["pgpass"] = pgpass_meta
    if not pgpass_meta.get("exists") or pgpass_meta.get("permissions") != "600":
        manifest["errors"].append("Expected ~/.pgpass to exist with permissions 600.")
        manifest_path = manifest_dir / "schema_audit.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return 3

    import wrds  # Imported only after preflight so local help/imports do not require WRDS.

    db = None
    try:
        db = wrds.Connection(wrds_username=username)
        manifest["wrds"]["connection_status"] = "ok"
        smoke = db.raw_sql("select 1 as ok")
        manifest["wrds"]["smoke_test_ok"] = bool(int(smoke.iloc[0]["ok"]) == 1)
        try:
            db.raw_sql("set statement_timeout to '120s'")
            manifest["wrds"]["statement_timeout"] = "120s"
        except Exception as exc:  # noqa: BLE001
            manifest["wrds"]["statement_timeout_warning"] = str(exc)

        libraries = sorted(db.list_libraries())
        manifest["wrds"]["library_count"] = len(libraries)
        candidate_libs_by_role: dict[str, list[str]] = {}
        all_candidate_libs: set[str] = set()
        for role, spec in ROLE_SPECS.items():
            role_libs = [lib for lib in libraries if library_score(lib, spec) > 0]
            if role.startswith("historical_segments"):
                role_libs = sorted(set(role_libs + [lib for lib in libraries if "seg" in lib.lower()]))
            candidate_libs_by_role[role] = role_libs
            all_candidate_libs.update(role_libs)

        table_inventory: dict[str, list[str]] = {}
        for lib in sorted(all_candidate_libs):
            try:
                tables = sorted(db.list_tables(library=lib))
                table_inventory[lib] = tables
            except Exception as exc:  # noqa: BLE001
                table_inventory[lib] = []
                manifest["errors"].append(f"Could not list tables for {lib}: {exc}")

        manifest["candidate_libraries"] = {
            role: {
                "libraries": libs,
                "table_counts": {lib: len(table_inventory.get(lib, [])) for lib in libs},
            }
            for role, libs in candidate_libs_by_role.items()
        }

        described_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

        def describe(library: str, table: str) -> list[dict[str, Any]]:
            key = (library, table)
            if key not in described_cache:
                described_cache[key] = normalize_columns(db.describe_table(library=library, table=table))
            return described_cache[key]

        def estimate_rows(library: str, table: str) -> int | None:
            sql = (
                "select c.reltuples::bigint as estimated_rows "
                "from pg_class c join pg_namespace n on n.oid = c.relnamespace "
                f"where n.nspname = {sql_literal(library)} and c.relname = {sql_literal(table)}"
            )
            try:
                rows = db.raw_sql(sql)
                if rows.empty:
                    return None
                value = rows.iloc[0]["estimated_rows"]
                if value is None:
                    return None
                return int(value)
            except Exception:
                return None

        def date_range(library: str, table: str, date_col: str) -> dict[str, Any] | None:
            sql = f"select min({qident(date_col)}) as min_date, max({qident(date_col)}) as max_date from {qident(library)}.{qident(table)}"
            try:
                rows = db.raw_sql(sql)
                if rows.empty:
                    return None
                return {"min": str(rows.iloc[0]["min_date"]), "max": str(rows.iloc[0]["max_date"])}
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc)}

        schema_map: dict[str, Any] = {
            "created_utc": manifest["created_utc"],
            "run_id": args.run_id,
            "project_root": str(project_root),
            "roles": {},
        }

        for role, spec in ROLE_SPECS.items():
            rough_candidates: list[dict[str, Any]] = []
            for lib in candidate_libs_by_role[role]:
                for table in table_inventory.get(lib, []):
                    base_score = library_score(lib, spec) + table_name_score(table, spec)
                    if base_score <= 0:
                        continue
                    rough_candidates.append({"library": lib, "table": table, "base_score": base_score})
            rough_candidates = sorted(rough_candidates, key=lambda x: (-x["base_score"], x["library"], x["table"]))[:12]

            inspected: list[dict[str, Any]] = []
            for cand in rough_candidates[:8]:
                try:
                    columns = describe(cand["library"], cand["table"])
                    col_score, hits, missing = column_score(columns, spec)
                    date_col = pick_date_column(columns, spec)
                    inspected.append(
                        {
                            "library": cand["library"],
                            "table": cand["table"],
                            "score": cand["base_score"] + col_score,
                            "base_score": cand["base_score"],
                            "column_count": len(columns),
                            "required_hits": hits,
                            "missing_required_groups": missing,
                            "date_column": date_col,
                            "columns": [c["name"] for c in columns[:80]],
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    inspected.append(
                        {
                            "library": cand["library"],
                            "table": cand["table"],
                            "score": cand["base_score"],
                            "error": str(exc),
                        }
                    )

            inspected = sorted(inspected, key=lambda x: (-int(x.get("score", 0)), x.get("library", ""), x.get("table", "")))
            selected = inspected[0] if inspected else None
            risks: list[str] = []
            if not selected:
                risks.append("No candidate table matched library/table-name heuristics.")
            elif selected.get("missing_required_groups"):
                risks.append(f"Missing required column groups: {selected['missing_required_groups']}")
            elif role == "benchmark_factors_wrds":
                risks.append("WRDS-hosted factor tables are optional; Kenneth French public files remain acceptable.")

            selected_public: dict[str, Any] | None = None
            if selected:
                selected_public = {k: v for k, v in selected.items() if k != "columns"}
                selected_public["estimated_rows"] = estimate_rows(selected["library"], selected["table"])
                if selected.get("date_column"):
                    selected_public["date_range"] = date_range(selected["library"], selected["table"], selected["date_column"])
                selected_public["columns_sample"] = selected.get("columns", [])[:40]

            role_entry = {
                "description": spec["description"],
                "selected": selected_public,
                "alternatives": [
                    {k: v for k, v in cand.items() if k != "columns"} for cand in inspected[1:6]
                ],
                "risks": risks,
            }
            schema_map["roles"][role] = role_entry
            manifest["roles"][role] = role_entry

        schema_path = config_dir / "schema_map.yml"
        manifest_path = manifest_dir / "schema_audit.json"
        report_path = report_dir / "schema_audit_report.md"

        schema_path.write_text(to_yaml(schema_map) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        report_path.write_text(markdown_report(manifest, schema_map), encoding="utf-8")

        print(f"schema_map={schema_path}")
        print(f"manifest={manifest_path}")
        print(f"report={report_path}")
        print("selected_roles=")
        for role, entry in schema_map["roles"].items():
            sel = entry.get("selected") or {}
            table = f"{sel.get('library')}.{sel.get('table')}" if sel else "UNRESOLVED"
            print(f"  {role}: {table}")
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest["wrds"]["connection_status"] = manifest["wrds"].get("connection_status", "failed")
        manifest["errors"].append(str(exc))
        manifest_path = manifest_dir / "schema_audit.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
