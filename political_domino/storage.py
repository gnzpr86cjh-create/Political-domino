"""Local persistence: one JSON file per assessment in ``data/assessments``."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from . import config

_ID = re.compile(r"^[a-f0-9]{10}$")


def _dir() -> Path:
    path = Path(config.DATA_DIR) / "assessments"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _path(assessment_id: str) -> Path:
    if not _ID.match(assessment_id or ""):
        raise ValueError("Invalid assessment id.")
    return _dir() / f"{assessment_id}.json"


def new_assessment(name: str, company: str, question: str, **extra) -> dict:
    a = {
        "schema_version": 1,
        "id": uuid.uuid4().hex[:10],
        "name": name.strip(),
        "company": company.strip(),
        "question": question.strip(),
        "industry": "",
        "market": "",
        "horizon": "",
        "background": "",
        "pestle": [],
        "initial_events": "",
        "created_at": _now(),
        "updated_at": _now(),
        "sources": [],
        "next_source_no": 1,
        "assessment_text": "",
        "assessment_origin": "",
        "analyst_map": {"nodes": [], "edges": [], "risk_factors": []},
        "manual_nodes": [],
        "manual_edges": [],
        "runs": [],
        "suggestions": [],
    }
    a.update(extra)
    save(a)
    return a


def save(a: dict) -> None:
    a["updated_at"] = _now()
    path = _path(a["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load(assessment_id: str) -> dict | None:
    try:
        path = _path(assessment_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        a = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    for key, default in (
        ("sources", []), ("runs", []), ("suggestions", []), ("manual_nodes", []),
        ("manual_edges", []), ("pestle", []), ("next_source_no", len(a.get("sources", [])) + 1),
        ("analyst_map", {"nodes": [], "edges": [], "risk_factors": []}),
    ):
        a.setdefault(key, default)
    return a


def list_assessments() -> list[dict]:
    items = []
    for path in _dir().glob("*.json"):
        a = load(path.stem)
        if a is None:
            continue
        decisions = [s.get("decision") for s in a["suggestions"]]
        items.append(
            {
                "id": a["id"], "name": a.get("name", "(unnamed)"), "company": a.get("company", ""),
                "updated_at": a.get("updated_at", ""), "n_sources": len(a["sources"]),
                "n_suggestions": len(decisions), "n_accepted": decisions.count("accepted"),
                "n_pending": decisions.count("pending"),
            }
        )
    return sorted(items, key=lambda i: i["updated_at"], reverse=True)


def delete(assessment_id: str) -> None:
    path = _path(assessment_id)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------- sources ----
def add_source(a: dict, extracted: dict) -> dict:
    """Attach an extracted file to the assessment (replacing a same-named one)."""
    a["sources"] = [s for s in a["sources"] if s["filename"] != extracted["filename"]]
    source = dict(extracted)
    source["id"] = f"S{a['next_source_no']}"
    source["added_at"] = _now()
    a["next_source_no"] += 1
    a["sources"].append(source)
    return source


def remove_source(a: dict, source_id: str) -> None:
    a["sources"] = [s for s in a["sources"] if s["id"] != source_id]


# ------------------------------------------------------------ suggestions ----
def get_suggestion(a: dict, suggestion_id: str) -> dict | None:
    return next((s for s in a["suggestions"] if s["id"] == suggestion_id), None)


def set_decision(a: dict, suggestion_id: str, decision: str) -> None:
    if decision not in config.DECISIONS:
        raise ValueError(f"Unknown decision: {decision}")
    s = get_suggestion(a, suggestion_id)
    if s is None:
        raise KeyError(suggestion_id)
    s["decision"] = decision
    s["decided_at"] = None if decision == "pending" else _now()


def record_run(a: dict, result: dict) -> None:
    """Store a finished analysis run. Earlier runs and decisions are kept as history."""
    a["analyst_map"] = result["analyst_map"]
    a["runs"].append(result["run"])
    a["suggestions"].extend(result["suggestions"])
