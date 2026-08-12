# SPDX-License-Identifier: MIT
"""Read/write gflow-director workflow files.

Mirrors ``ui/src-tauri/src/workflow.rs`` byte-for-byte: same directory layout
(``<app_data_dir>/projects/<project_id>/<workflow_id>.json``), same
``WorkflowFile`` JSON shape (``{id, name, updatedAt, nodes, edges}``), same
"the directory listing IS the index" convention. This lets MCP-created/run
workflows show up live in the real gflow-director GUI and vice versa — one
source of truth for both clients.

The Rust sibling never validates ``nodes``/``edges`` shape ("the graph shape
belongs to the frontend, not the backend") — this module keeps that same
discipline: nodes/edges pass through as opaque JSON-compatible values.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

#: Must match the ``identifier`` in ui/src-tauri/tauri.conf.json — that value
#: is what Tauri's ``app.path().app_data_dir()`` keys off of.
GFLOW_DIRECTOR_IDENTIFIER = "com.mukesh.gflowdirector"


class WorkflowNotFoundError(Exception):
    """Raised by callers that need a hard failure instead of ``None``."""

    def __init__(self, project_id: str, workflow_id: str) -> None:
        super().__init__(f"workflow {workflow_id!r} not found in project {project_id!r}")
        self.project_id = project_id
        self.workflow_id = workflow_id


@dataclass
class WorkflowSummary:
    id: str
    name: str
    updated_at: str


def _empty_list() -> list[Any]:
    return []


@dataclass
class WorkflowFile:
    id: str
    name: str
    updated_at: str
    nodes: list[Any] = field(default_factory=_empty_list)
    edges: list[Any] = field(default_factory=_empty_list)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "updatedAt": self.updated_at,
            "nodes": self.nodes,
            "edges": self.edges,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> WorkflowFile:
        return cls(
            id=data["id"],
            name=data["name"],
            updated_at=data["updatedAt"],
            nodes=data.get("nodes", []),
            edges=data.get("edges", []),
        )


def app_data_dir() -> Path:
    """gflow-director's Tauri app-data directory.

    Mirrors Tauri's ``app.path().app_data_dir()`` resolution for identifier
    ``com.mukesh.gflowdirector``: ``~/.local/share/<id>`` on Linux,
    ``~/Library/Application Support/<id>`` on macOS, ``%APPDATA%\\<id>`` on
    Windows (Tauri's Rust ``dirs`` crate uses the *roaming* app-data folder,
    hence ``roaming=True`` here). Override with ``GFLOW_DIRECTOR_APP_DATA_DIR``
    (e.g. for tests, or to point at another machine's synced app-data copy).
    """
    override = os.environ.get("GFLOW_DIRECTOR_APP_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path(user_data_dir(GFLOW_DIRECTOR_IDENTIFIER, appauthor=False, roaming=True))


def projects_root() -> Path:
    return app_data_dir() / "projects"


def project_dir(project_id: str) -> Path:
    return projects_root() / project_id


def app_settings_path() -> Path:
    return app_data_dir() / "app_settings.json"


def active_project_id() -> str | None:
    """Best-effort read of the GUI's ``app_settings.json`` ``activeProjectId``.

    Returns ``None`` if the file doesn't exist or doesn't parse — callers
    should require an explicit ``project_id`` in that case rather than
    guessing one.
    """
    path = app_settings_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    project_id = data.get("activeProjectId")
    return project_id if isinstance(project_id, str) else None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _workflow_path(project_id: str, workflow_id: str) -> Path:
    return project_dir(project_id) / f"{workflow_id}.json"


def list_workflows(project_id: str) -> list[WorkflowSummary]:
    """List every workflow under a project, newest-updated first.

    Every ``<workflow_id>.json`` under the project directory IS the index —
    no separate manifest to keep in sync. Skips any file that doesn't parse
    rather than failing the whole list (matches Rust ``workflow::list``).
    """
    directory = project_dir(project_id)
    if not directory.exists():
        return []
    out: list[WorkflowSummary] = []
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            out.append(
                WorkflowSummary(id=data["id"], name=data["name"], updated_at=data["updatedAt"])
            )
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    out.sort(key=lambda w: w.updated_at, reverse=True)
    return out


def get_workflow(project_id: str, workflow_id: str) -> WorkflowFile | None:
    path = _workflow_path(project_id, workflow_id)
    if not path.exists():
        return None
    return WorkflowFile.from_json_dict(json.loads(path.read_text()))


def create_workflow(project_id: str, name: str) -> WorkflowFile:
    wf = WorkflowFile(id=str(uuid.uuid4()), name=name, updated_at=_now(), nodes=[], edges=[])
    _write(project_id, wf)
    return wf


def save_workflow(
    project_id: str,
    workflow_id: str,
    name: str,
    nodes: list[Any],
    edges: list[Any],
) -> WorkflowFile:
    """Overwrite the file, refreshing ``updatedAt``.

    ``nodes``/``edges`` pass through untouched, matching Rust ``save()``'s
    "the backend doesn't interpret the graph shape" discipline.
    """
    wf = WorkflowFile(id=workflow_id, name=name, updated_at=_now(), nodes=nodes, edges=edges)
    _write(project_id, wf)
    return wf


def delete_workflow(project_id: str, workflow_id: str) -> None:
    path = _workflow_path(project_id, workflow_id)
    if path.exists():
        path.unlink()


def _write(project_id: str, wf: WorkflowFile) -> None:
    directory = project_dir(project_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{wf.id}.json"
    path.write_text(json.dumps(wf.to_json_dict(), indent=2))


# ---------------------------------------------------------------------------
# Dependency resolution — pure functions mirroring ui/src/store/graphStore.ts
# ---------------------------------------------------------------------------
#
# The only two target handles any node type in gflow-director consumes are
# 'prompt' (Text -> Image/Video) and 'initialFrame' (Image -> Video); see
# ImageNode.tsx/VideoNode.tsx/TextNode.tsx's <Handle> ids. CharacterNode has
# no handles at all — it's referenced by name via '@Name' text mentions,
# resolved by gflow itself server-side, never via an edge.

_DEPENDENCY_HANDLES = ("prompt", "initialFrame")

#: Node types whose 'text' source handle produces prose usable as an
#: upstream 'prompt' input — the original Text node plus the two Codex-CLI
#: writer nodes, which store their generated output in the same `data.text`
#: field. Mirrors graphStore.ts::getUpstreamText's widened type check.
_TEXT_SOURCE_TYPES = ("text", "characterWriter", "locationWriter")

#: Node types whose 'image' source handle produces a usable upstream image
#: path — the original Image node plus the two chatgpt.com-backed sheet
#: nodes. Mirrors graphStore.ts::getUpstreamImagePath's widened type check.
_IMAGE_SOURCE_TYPES = ("image", "characterSheet", "locationSheet")


def topological_order(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str] | None:
    """Return node ids in an order where every dependency runs before its
    dependents, or ``None`` if the dependency edges form a cycle.

    Only edges whose ``targetHandle`` is 'prompt' or 'initialFrame' count as
    dependencies — those are the only two handles this app's nodes have.
    """
    node_ids = [n["id"] for n in nodes]
    node_id_set = set(node_ids)
    depends_on: dict[str, set[str]] = {nid: set() for nid in node_ids}
    dependents: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for edge in edges:
        if edge.get("targetHandle") not in _DEPENDENCY_HANDLES:
            continue
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if source in node_id_set and target in node_id_set and source != target:
            depends_on[target].add(source)
            dependents[source].add(target)

    remaining = {nid: len(deps) for nid, deps in depends_on.items()}
    ready = [nid for nid, count in remaining.items() if count == 0]
    order: list[str] = []
    while ready:
        nid = ready.pop()
        order.append(nid)
        for dependent in dependents[nid]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)

    if len(order) != len(node_ids):
        return None  # a cycle left some node's in-degree above zero forever
    return order


def upstream_text(
    node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str | None:
    """The text feeding *node_id*'s 'prompt' handle, or ``None`` if there's no
    such edge or its source isn't a Text/Character-Writer/Location-Writer
    node. Mirrors ``graphStore.ts::getUpstreamText``."""
    for edge in edges:
        source_id = edge.get("source")
        if not isinstance(source_id, str):
            continue
        if edge.get("target") == node_id and edge.get("targetHandle") == "prompt":
            source = nodes_by_id.get(source_id)
            if source is not None and source.get("type") in _TEXT_SOURCE_TYPES:
                text = source.get("data", {}).get("text")
                return text if isinstance(text, str) else None
    return None


def upstream_image_path(
    node_id: str,
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str | None:
    """The image path feeding *node_id*'s 'initialFrame' handle, or ``None`` if
    there's no such edge or its source isn't an Image/Character-Sheet/
    Location-Sheet node. Prefers a
    user-imported ``localFilePath`` over a generated ``artifactPath``, same
    priority as ``graphStore.ts::getUpstreamImagePath`` (minus its in-memory
    ``runStore`` lookup, which has nothing to read outside a live GUI session —
    ``artifactPath`` is what survives a restart anyway, so it's the faithful
    match for anything this module can see)."""
    for edge in edges:
        source_id = edge.get("source")
        if not isinstance(source_id, str):
            continue
        if edge.get("target") == node_id and edge.get("targetHandle") == "initialFrame":
            source = nodes_by_id.get(source_id)
            if source is not None and source.get("type") in _IMAGE_SOURCE_TYPES:
                data = source.get("data", {})
                local = data.get("localFilePath")
                if isinstance(local, str) and local:
                    return local
                artifact = data.get("artifactPath")
                return artifact if isinstance(artifact, str) else None
    return None
