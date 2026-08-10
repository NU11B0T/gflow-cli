# SPDX-License-Identifier: MIT
"""Unit tests for gflow_cli.mcp.workflow_files.

Covers the file-based WorkflowFile CRUD (isolated to a tmp_path app-data dir
via GFLOW_DIRECTOR_APP_DATA_DIR) and the pure dependency-resolution helpers
(topological_order, upstream_text, upstream_image_path) that
gflow_workflow_run relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gflow_cli.mcp import workflow_files as wf


@pytest.fixture(autouse=True)
def _isolated_app_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GFLOW_DIRECTOR_APP_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# app_data_dir / active_project_id
# ---------------------------------------------------------------------------


def test_app_data_dir_honors_env_override(tmp_path: Path) -> None:
    assert wf.app_data_dir() == tmp_path


def test_active_project_id_missing_file_returns_none() -> None:
    assert wf.active_project_id() is None


def test_active_project_id_reads_app_settings(tmp_path: Path) -> None:
    (tmp_path / "app_settings.json").write_text('{"activeProjectId": "PROJ-abc123"}')
    assert wf.active_project_id() == "PROJ-abc123"


def test_active_project_id_bad_json_returns_none(tmp_path: Path) -> None:
    (tmp_path / "app_settings.json").write_text("not json")
    assert wf.active_project_id() is None


# ---------------------------------------------------------------------------
# WorkflowFile CRUD
# ---------------------------------------------------------------------------


def test_create_then_get_roundtrips() -> None:
    created = wf.create_workflow("PROJ-1", "My Workflow")
    fetched = wf.get_workflow("PROJ-1", created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "My Workflow"
    assert fetched.nodes == []
    assert fetched.edges == []


def test_get_nonexistent_returns_none() -> None:
    assert wf.get_workflow("PROJ-1", "does-not-exist") is None


def test_save_overwrites_and_bumps_updated_at() -> None:
    created = wf.create_workflow("PROJ-1", "Name 1")
    first_updated_at = created.updated_at

    nodes = [
        {"id": "text_1", "type": "text", "data": {"text": "hello"}, "position": {"x": 0, "y": 0}}
    ]
    saved = wf.save_workflow("PROJ-1", created.id, "Name 2", nodes, [])

    assert saved.id == created.id
    assert saved.name == "Name 2"
    assert saved.nodes == nodes
    assert saved.updated_at >= first_updated_at

    fetched = wf.get_workflow("PROJ-1", created.id)
    assert fetched is not None
    assert fetched.name == "Name 2"
    assert fetched.nodes == nodes


def test_save_writes_the_exact_json_shape_the_gui_reads(tmp_path: Path) -> None:
    saved = wf.save_workflow("PROJ-1", "wf-123", "Workflow 1", [], [])
    path = tmp_path / "projects" / "PROJ-1" / "wf-123.json"
    assert path.exists()

    import json

    on_disk = json.loads(path.read_text())
    assert on_disk == {
        "id": "wf-123",
        "name": "Workflow 1",
        "updatedAt": saved.updated_at,
        "nodes": [],
        "edges": [],
    }


def test_list_workflows_sorted_newest_first() -> None:
    a = wf.create_workflow("PROJ-1", "A")
    b = wf.create_workflow("PROJ-1", "B")
    # Force a strictly later timestamp than 'a' so ordering is unambiguous
    # even if both creates land in the same microsecond on a fast machine.
    wf.save_workflow("PROJ-1", b.id, "B", [], [])

    summaries = wf.list_workflows("PROJ-1")
    ids_in_order = [s.id for s in summaries]
    assert ids_in_order[0] == b.id
    assert a.id in ids_in_order


def test_list_workflows_empty_project_returns_empty_list() -> None:
    assert wf.list_workflows("PROJ-nonexistent") == []


def test_list_workflows_skips_unparseable_file(tmp_path: Path) -> None:
    wf.create_workflow("PROJ-1", "Good")
    (tmp_path / "projects" / "PROJ-1" / "garbage.json").write_text("{not json")

    summaries = wf.list_workflows("PROJ-1")
    assert len(summaries) == 1
    assert summaries[0].name == "Good"


def test_delete_workflow_removes_file() -> None:
    created = wf.create_workflow("PROJ-1", "Temp")
    wf.delete_workflow("PROJ-1", created.id)
    assert wf.get_workflow("PROJ-1", created.id) is None


def test_delete_nonexistent_workflow_is_a_noop() -> None:
    wf.delete_workflow("PROJ-1", "never-existed")  # must not raise


# ---------------------------------------------------------------------------
# topological_order
# ---------------------------------------------------------------------------


def _node(node_id: str, node_type: str, data: dict | None = None) -> dict:
    return {"id": node_id, "type": node_type, "data": data or {}, "position": {"x": 0, "y": 0}}


def _edge(source: str, target: str, target_handle: str, source_handle: str = "out") -> dict:
    return {
        "id": f"{source}-{target}",
        "source": source,
        "target": target,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
    }


def test_topological_order_linear_chain() -> None:
    nodes = [_node("t1", "text"), _node("i1", "image"), _node("v1", "video")]
    edges = [
        _edge("t1", "i1", "prompt"),
        _edge("t1", "v1", "prompt"),
        _edge("i1", "v1", "initialFrame"),
    ]

    order = wf.topological_order(nodes, edges)
    assert order is not None
    assert order.index("t1") < order.index("i1") < order.index("v1")


def test_topological_order_disconnected_nodes_all_included() -> None:
    nodes = [_node("t1", "text"), _node("c1", "character")]
    order = wf.topological_order(nodes, [])
    assert order is not None
    assert set(order) == {"t1", "c1"}


def test_topological_order_detects_cycle() -> None:
    nodes = [_node("i1", "image"), _node("v1", "video")]
    # Not realistic for this app's node types, but exercises cycle detection
    # generically: i1 depends on v1's prompt, v1 depends on i1's initialFrame.
    edges = [_edge("v1", "i1", "prompt"), _edge("i1", "v1", "initialFrame")]
    assert wf.topological_order(nodes, edges) is None


def test_topological_order_ignores_non_dependency_handles() -> None:
    nodes = [_node("a", "text"), _node("b", "text")]
    edges = [_edge("a", "b", "some-other-handle")]
    order = wf.topological_order(nodes, edges)
    assert order is not None
    assert set(order) == {"a", "b"}


# ---------------------------------------------------------------------------
# upstream_text / upstream_image_path
# ---------------------------------------------------------------------------


def test_upstream_text_resolves_from_connected_text_node() -> None:
    nodes = [_node("t1", "text", {"text": "a cat"}), _node("i1", "image")]
    edges = [_edge("t1", "i1", "prompt")]
    nodes_by_id = {n["id"]: n for n in nodes}
    assert wf.upstream_text("i1", nodes_by_id, edges) == "a cat"


def test_upstream_text_none_when_no_edge() -> None:
    nodes = [_node("i1", "image")]
    nodes_by_id = {n["id"]: n for n in nodes}
    assert wf.upstream_text("i1", nodes_by_id, []) is None


def test_upstream_text_none_when_source_is_not_text() -> None:
    nodes = [_node("i1", "image"), _node("i2", "image")]
    edges = [_edge("i1", "i2", "prompt")]
    nodes_by_id = {n["id"]: n for n in nodes}
    assert wf.upstream_text("i2", nodes_by_id, edges) is None


def test_upstream_image_path_prefers_local_file_over_artifact() -> None:
    nodes = [
        _node("i1", "image", {"localFilePath": "/local/x.png", "artifactPath": "/artifact/x.png"}),
        _node("v1", "video"),
    ]
    edges = [_edge("i1", "v1", "initialFrame")]
    nodes_by_id = {n["id"]: n for n in nodes}
    assert wf.upstream_image_path("v1", nodes_by_id, edges) == "/local/x.png"


def test_upstream_image_path_falls_back_to_artifact_path() -> None:
    nodes = [
        _node("i1", "image", {"localFilePath": None, "artifactPath": "/artifact/x.png"}),
        _node("v1", "video"),
    ]
    edges = [_edge("i1", "v1", "initialFrame")]
    nodes_by_id = {n["id"]: n for n in nodes}
    assert wf.upstream_image_path("v1", nodes_by_id, edges) == "/artifact/x.png"


def test_upstream_image_path_none_when_no_edge() -> None:
    nodes = [_node("v1", "video")]
    nodes_by_id = {n["id"]: n for n in nodes}
    assert wf.upstream_image_path("v1", nodes_by_id, []) is None
