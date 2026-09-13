"""Small durable architecture contract and pragmatic consistency checks."""

from __future__ import annotations

from pathlib import Path


def default_contract(specification: str, environment: dict[str, object]) -> dict[str, str]:
    text = specification.lower()
    npm = bool(((environment.get("npm") or {}) if isinstance(environment.get("npm"), dict) else {}).get("available"))
    if any(token in text for token in ("python", "flask", "sqlite", "backend", "бэкенд")):
        return {
            "backend_framework": "Flask",
            "orm": "Flask-SQLAlchemy",
            "database": "SQLite",
            "frontend": "npm-build" if npm else "static-html-css-js",
            "test_framework": "unittest",
        }
    return {}


def validate_architecture(workspace: Path, contract: dict[str, object]) -> list[str]:
    if not contract:
        return []
    source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in workspace.rglob("*.py")
        if ".venv" not in path.parts and ".autodev" not in path.parts
    ).lower()
    findings: list[str] = []
    uses_peewee = "import peewee" in source or "from peewee" in source
    uses_sqlalchemy = "sqlalchemy" in source
    if uses_peewee and uses_sqlalchemy:
        findings.append("Architecture conflict: conflicting ORMs Peewee and SQLAlchemy are both imported")
    selected_orm = str(contract.get("orm", "")).lower()
    if selected_orm == "flask-sqlalchemy" and uses_peewee:
        findings.append("Architecture conflict: contract selects Flask-SQLAlchemy but source imports Peewee")
    if selected_orm == "peewee" and uses_sqlalchemy:
        findings.append("Architecture conflict: contract selects Peewee but source imports SQLAlchemy")
    frontend = str(contract.get("frontend", ""))
    if frontend == "static-html-css-js" and any(workspace.rglob("package.json")):
        findings.append("Architecture conflict: static frontend contract conflicts with package.json")
    requirements = workspace / "requirements.txt"
    if selected_orm == "flask-sqlalchemy" and requirements.is_file() and "flask-sqlalchemy" not in requirements.read_text(encoding="utf-8", errors="ignore").lower():
        findings.append("Architecture conflict: Flask-SQLAlchemy import is absent from requirements.txt")
    return findings
