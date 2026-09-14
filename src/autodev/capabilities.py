"""Durable project contracts and one-time capability-plan normalization.

The controller deliberately works with stable capability identifiers rather
than task titles.  Titles are model-facing prose and therefore cannot be used
as durable identity across language changes, resume, or corrective work.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .models import TaskStatus

if TYPE_CHECKING:
    from .models import ProjectState, Task

CapabilityGraph = dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class ContractPolicyViolation:
    """A deterministic rejection of lower-authority generated code.

    The project contract is chosen before implementation.  A later model
    action must not be able to silently replace that decision by importing a
    framework or asking the environment layer to install one.
    """

    code: str
    message: str
    subject: str


def build_project_contract(specification: str, environment: dict[str, object]) -> dict[str, object]:
    """Create the compact, versioned contract selected before implementation."""
    text = specification.lower()
    npm = bool((environment.get("npm") if isinstance(environment.get("npm"), dict) else {}).get("available"))
    python_project = any(token in text for token in ("python", "flask", "sqlite", "backend", "бэкенд"))
    frontend = "npm-build" if npm else "static-html-css-js"
    stdlib_only = _requests_stdlib_only(text)
    forbidden_dependencies = _explicit_forbidden_dependencies(text)
    architecture: dict[str, object] = {
        "backend_framework": "stdlib-http" if python_project and stdlib_only else "Flask" if python_project else "unspecified",
        "database": "SQLite" if "sqlite" in text else "unspecified",
        "frontend_strategy": frontend,
        "app_factory": False,
        "test_strategy": "unittest" if python_project else "unspecified",
        "runtime_behavior": "managed-process-no-reloader",
    }
    forbidden = ["package.json", "npm build requirement"] if frontend == "static-html-css-js" else []
    return {
        "version": 1,
        "mode": "reliable",
        "architecture": architecture,
        "api": {},
        "data": {},
        "capabilities": {},
        "technology_constraints": {
            "stdlib_only_python": stdlib_only,
            "forbidden_dependencies": forbidden_dependencies,
            "allowed_dependencies": [] if stdlib_only else (["Flask"] if python_project else []),
        },
        "forbidden_requirements": forbidden,
        "original_spec_hash": _signature(specification),
    }


def contract_policy_violation(contract: dict[str, object], path: str, content: str, workspace: object | None = None) -> ContractPolicyViolation | None:
    """Detect source text that violates an explicit technology constraint.

    This deliberately works on imports rather than a Notes-specific framework
    list.  Project-local modules and Python's standard library remain valid;
    third-party imports are forbidden only when the frozen contract says so.
    """
    constraints = contract.get("technology_constraints")
    if not isinstance(constraints, dict):
        return None
    stdlib_only = constraints.get("stdlib_only_python") is True
    forbidden = {
        str(name).strip().lower().replace("-", "_")
        for name in constraints.get("forbidden_dependencies", [])
        if isinstance(name, str)
    }
    if not stdlib_only and not forbidden:
        return None
    for module in _imported_roots(content):
        normalized = module.lower().replace("-", "_")
        if _is_stdlib_module(normalized) or _is_workspace_module(workspace, normalized):
            continue
        if stdlib_only or normalized in forbidden:
            return ContractPolicyViolation(
                "CONTRACT_POLICY_VIOLATION",
                f"Frozen project contract forbids external dependency '{module}' in {path}",
                module,
            )
    return None


def contract_allows_dependency(contract: dict[str, object] | None, dependency: str) -> bool:
    """Return whether the owned project environment may install *dependency*."""
    if not isinstance(contract, dict):
        return True
    constraints = contract.get("technology_constraints")
    if not isinstance(constraints, dict):
        return True
    normalized = dependency.strip().lower().replace("-", "_")
    if constraints.get("stdlib_only_python") is True:
        return False
    forbidden = {
        str(name).strip().lower().replace("-", "_")
        for name in constraints.get("forbidden_dependencies", [])
        if isinstance(name, str)
    }
    return normalized not in forbidden


def contract_declares_dependency(contract: dict[str, object] | None, dependency: str) -> bool:
    """Whether the frozen contract explicitly models a non-manifest package."""
    if not isinstance(contract, dict):
        return True
    constraints = contract.get("technology_constraints")
    if not isinstance(constraints, dict):
        return True
    allowed = constraints.get("allowed_dependencies", [])
    if not isinstance(allowed, list):
        return False
    normalized = dependency.strip().lower().replace("-", "_")
    return any(isinstance(item, str) and item.strip().lower().replace("-", "_") == normalized for item in allowed)


def find_contract_policy_violations(contract: dict[str, object], workspace: object) -> list[tuple[str, ContractPolicyViolation]]:
    """Scan owned source files before environment repair can install anything.

    This is intentionally a small import-policy scan, not a linter.  Its only
    job is to prevent a generated forbidden framework from acquiring authority
    merely because its import fails during validation.
    """
    try:
        root = __import__("pathlib").Path(workspace)
    except TypeError:
        return []
    findings: list[tuple[str, ContractPolicyViolation]] = []
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", ".autodev", "__pycache__"} for part in path.parts):
            continue
        try:
            relative = path.relative_to(root).as_posix()
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        violation = contract_policy_violation(contract, relative, content, root)
        if violation is not None:
            findings.append((relative, violation))
    return findings


def capability_id_for(task: "Task") -> str:
    """Return a stable generic identity, preferring planner-supplied intent.

    The fallback is a conservative normalized intent; it never expands tasks
    or fabricates product-specific work.  A manager can supply capability_id
    explicitly in future structured plans.
    """
    if task.capability_id:
        return task.capability_id
    text = f"{task.title} {task.description}".lower()
    # These are implementation-independent product verbs, not product names.
    # They allow a bilingual manager plan to converge on one canonical owner.
    if any(word in text for word in ("favorite", "favourite", "избран")):
        return "domain.favorites"
    if any(word in text for word in ("categor", "категор")):
        return "domain.categories"
    if any(word in text for word in ("search", "поиск", "filter", "фильтр")):
        return "domain.search-filter"
    if any(word in text for word in ("validation", "валидац", "error handling", "обработк ошибок")):
        return "domain.validation-errors"
    verbs = sum(word in text for word in ("crud", "create", "update", "edit", "delete", "создан", "редакт", "удален", "удалён"))
    if verbs >= 2:
        return "domain.crud"
    if any(word in text for word in ("responsive", "style", "css", "адаптив", "стил")):
        return "frontend.responsive"
    if any(word in text for word in ("javascript", "actions", "interaction", "действ", "интерактив")):
        return "frontend.actions"
    # Generic category only: avoids the previous Notes-specific atomizer.
    groups = (
        ("documentation", ("readme", "documentation", "документ")),
        ("tests", ("test", "pytest", "unittest", "тест")),
        ("frontend", ("frontend", "front-end", " ui", "html", "css", "интерфейс", "фронтенд")),
        ("persistence", ("database", "sqlite", "storage", "баз", "хранен")),
        ("backend", ("backend", "api", "server", "бэкенд", "сервер")),
    )
    category = next((name for name, markers in groups if any(marker in text for marker in markers)), "capability")
    # Keep descriptive words for a stable, readable fallback.  It is not
    # semantic translation; cross-language plans should carry an explicit ID.
    words = re.findall(r"[a-z0-9а-яё]+", text)
    meaningful = [word for word in words if len(word) > 2 and word not in {"implement", "create", "build", "реализовать", "создать", "задача"}]
    suffix = "-".join(meaningful[:5]) or _signature(text)[:10]
    return f"{category}.{suffix}"


def normalize_plan(state: "ProjectState") -> CapabilityGraph:
    """Canonicalize an already-created plan once without growing it on resume."""
    graph: CapabilityGraph = dict(state.capability_graph)
    canonical: dict[str, "Task"] = {}
    for task in state.tasks:
        if task.status is TaskStatus.SUPERSEDED:
            continue
        capability_id = capability_id_for(task)
        task.capability_id = capability_id
        incumbent = canonical.get(capability_id)
        if incumbent is None:
            canonical[capability_id] = task
            graph[capability_id] = {
                "task_id": task.id,
                "status": task.status.value,
                "intent": task.intent or task.description,
                "acceptance": list(task.acceptance_criteria),
                "contract_version": task.contract_version,
            }
            continue
        # Equivalent work belongs to the existing root. Preserve the work as
        # durable evidence, but never schedule a second independent owner.
        task.status = TaskStatus.SUPERSEDED
        task.root_task_id = incumbent.root_task_id
        task.parent_task_id = incumbent.id
        for criterion in task.acceptance_criteria:
            if criterion not in incumbent.acceptance_criteria:
                incumbent.acceptance_criteria.append(criterion)
        entry = graph[capability_id]
        entry["acceptance"] = list(incumbent.acceptance_criteria)
    state.capability_graph = graph
    if state.project_contract:
        contract_capabilities = state.project_contract.setdefault("capabilities", {})
        if isinstance(contract_capabilities, dict):
            for capability_id, entry in graph.items():
                contract_capabilities.setdefault(
                    capability_id,
                    {"intent": entry.get("intent", ""), "acceptance": entry.get("acceptance", [])},
                )
    return graph


def contract_conflicting_review_reasons(contract: dict[str, object], reasons: object) -> list[str]:
    """Return reviewer objections which demand a forbidden implementation.

    Reviewer feedback is lower authority than the accepted project contract.
    This intentionally filters only direct toolchain contradictions; ordinary
    quality findings remain reviewer-owned.
    """
    architecture = contract.get("architecture", {})
    frontend = architecture.get("frontend_strategy") if isinstance(architecture, dict) else ""
    if frontend != "static-html-css-js" or not isinstance(reasons, list):
        return []
    conflicts: list[str] = []
    for reason in reasons:
        if not isinstance(reason, str):
            continue
        lowered = reason.lower()
        if "package.json" in lowered or "npm" in lowered or "node" in lowered:
            conflicts.append(reason)
    return conflicts


def contract_conflicting_test_action(contract: dict[str, object], action: object) -> str | None:
    """Reject only direct test assertions that contradict a frozen contract."""
    if not isinstance(action, dict) or action.get("kind") not in {"write_file", "append_file", "edit_file"}:
        return None
    path = action.get("path")
    content = action.get("content") or action.get("new") or ""
    if not isinstance(path, str) or not isinstance(content, str):
        return None
    if "test" not in path.lower():
        return None
    architecture = contract.get("architecture", {})
    if not isinstance(architecture, dict):
        return None
    if architecture.get("app_factory") is False and "create_app" in content:
        return "TEST_CONTRACT_CONFLICT: contract selects a module-level application, but generated test requires create_app"
    return None


def repair_generated_test_for_contract(contract: dict[str, object], content: str) -> str | None:
    """Apply a mechanical lower-authority test repair for app-factory drift."""
    architecture = contract.get("architecture", {})
    if not isinstance(architecture, dict) or architecture.get("app_factory") is not False:
        return None
    symbol = str(architecture.get("application_symbol", "app"))
    if "create_app" not in content:
        return None
    # Preserve the module selected by the generated artifact.  Run-14 used
    # ``from backend.app import create_app, init_db``; forcing it to ``app.py``
    # silently changed a lower-authority test into a different application.
    # An alias has a different call site, so normalize that representation
    # before the generic direct-import repair below.
    repaired = content
    aliases = list(re.finditer(
        r"from\s+([A-Za-z_][\w.]*)\s+import\s+create_app\s+as\s+([A-Za-z_]\w*)",
        repaired,
    ))
    for match in aliases:
        module, alias = match.group(1), match.group(2)
        repaired = repaired.replace(match.group(0), f"from {module} import {symbol}", 1)
        repaired = re.sub(rf"\b{re.escape(alias)}\s*\(\s*\)", symbol, repaired)
    # Replace only the forbidden imported symbol and its direct call.
    repaired = re.sub(
        r"(from\s+[A-Za-z_][\w.]*\s+import\s+[^\n]*?)\bcreate_app\b",
        lambda match: match.group(1) + symbol,
        repaired,
    )
    repaired = re.sub(r"\bcreate_app\s*\(\s*\)", symbol, repaired)
    return repaired if repaired != content and "create_app" not in repaired else None


def reviewer_scope_violations(title: str, description: str, contract: dict[str, object], reasons: object) -> list[str]:
    """Reject reviewer demands for unrelated, future capability work."""
    if not isinstance(reasons, list):
        return []
    current = f"{title} {description}".lower()
    architecture = contract.get("architecture", {})
    entry = str(architecture.get("application_entry_point", "")) if isinstance(architecture, dict) else ""
    future_markers = {
        "frontend": ("frontend", "html", "css", "javascript", "static asset"),
        "documentation": ("readme", "documentation", "setup instruction"),
    }
    violations: list[str] = []
    for reason in reasons:
        if not isinstance(reason, str):
            continue
        lowered = reason.lower()
        if entry and "backend/app.py" in lowered and entry.lower() != "backend/app.py":
            violations.append(reason)
            continue
        for capability, markers in future_markers.items():
            if any(marker in lowered for marker in markers) and not any(marker in current for marker in markers):
                violations.append(reason)
                break
    return violations


def _signature(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def _requests_stdlib_only(text: str) -> bool:
    markers = (
        "standard library only",
        "python standard library only",
        "stdlib only",
        "no external python dependenc",
        "только стандартная библиотека",
        "без внешних python",
        "без внешних зависимостей python",
    )
    return any(marker in text for marker in markers)


def _explicit_forbidden_dependencies(text: str) -> list[str]:
    """Extract named, user-forbidden packages without inventing a framework list."""
    values: set[str] = set()
    for match in re.finditer(r"\b(?:no|without|exclude)\s+([a-z][a-z0-9_-]*(?:\s*,\s*[a-z][a-z0-9_-]*)*)", text):
        values.update(item.strip().replace("-", "_") for item in match.group(1).split(","))
    return sorted(values)


def _imported_roots(content: str) -> list[str]:
    roots: list[str] = []
    for line in content.splitlines():
        match = re.match(r"\s*import\s+([A-Za-z_]\w*)", line)
        if match:
            roots.append(match.group(1))
            continue
        match = re.match(r"\s*from\s+([A-Za-z_]\w*)", line)
        if match:
            roots.append(match.group(1))
    return roots


def _is_stdlib_module(module: str) -> bool:
    return module in getattr(sys, "stdlib_module_names", set()) or module in sys.builtin_module_names


def _is_workspace_module(workspace: object | None, module: str) -> bool:
    if workspace is None:
        return False
    try:
        root = __import__("pathlib").Path(workspace)
    except TypeError:
        return False
    candidate = root / module
    return candidate.is_dir() or candidate.with_suffix(".py").is_file()
