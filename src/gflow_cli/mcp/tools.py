# SPDX-License-Identifier: MIT
"""MCP tool definitions — maps MCP tools to gflow-cli core functions.

Each tool is registered on the shared MCPServer instance and delegates
to FlowWorker / DataStore / data.queries for actual execution.

Rate limiting: a token-bucket (capacity=8, refill=1/20s) prevents runaway
agentic loops from burning credits. Session and daily budget limits are
enforced by checking spent amounts against SQLite records.

Task claiming: the direct-execution path enqueues a task then claims it via
the atomic ``QueueRepository.claim_task`` (the same BEGIN IMMEDIATE claim the
daemon poll loop uses), so the two paths can never execute the same row. The
former per-profile asyncio lock is gone — serialising concurrent same-profile
BROWSER sessions is the browser lease's job (production-readiness plan slice
D); until it lands, two distinct same-profile tasks may drive two browsers
concurrently (the claim still prevents double-executing one task).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import structlog

from gflow_cli._cli_helpers import _FLOW_ID_RE
from gflow_cli.api.character import VOICES, CharacterImageRequest
from gflow_cli.api.client import FlowApiClient
from gflow_cli.api.image import AgentInstruction
from gflow_cli.api.video import is_media_uuid
from gflow_cli.cli_instructions import classify_refs
from gflow_cli.config import UiMode, get_settings
from gflow_cli.data.models import AssetLookup
from gflow_cli.data.queries import list_projects
from gflow_cli.data.recorder import OperationRecorder
from gflow_cli.data.repository import DataRepository
from gflow_cli.data.store import DataStore
from gflow_cli.errors import GFlowError, is_retryable
from gflow_cli.mcp.server import server
from gflow_cli.mcp.workflow_files import (
    active_project_id,
    get_workflow,
    list_workflows,
    save_workflow,
    topological_order,
    upstream_image_path,
    upstream_text,
)
from gflow_cli.profile_store import NoDefaultProfileError, NoProfilesError, resolve_profile
from gflow_cli.services.character_create import character_create
from gflow_cli.worker import codec
from gflow_cli.worker.daemon import FlowWorker
from gflow_cli.worker.queue import QueueRepository

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Token bucket rate limiter
# ---------------------------------------------------------------------------

_BUCKET_CAPACITY = 8
_BUCKET_REFILL_RATE = 1 / 20  # 1 token every 20 seconds


class _TokenBucket:
    """Simple token-bucket rate limiter for generation tools."""

    def __init__(self, capacity: int = _BUCKET_CAPACITY, refill_rate: float = _BUCKET_REFILL_RATE):
        self._capacity = capacity
        self._tokens = float(capacity)
        self._refill_rate = refill_rate
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a token. Returns True if acquired, False if rate-limited."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last_refill = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


_rate_limiter = _TokenBucket()

# Sentinel meaning "auto-resolve the profile like the CLI" (see
# _resolve_and_validate_profile). Shared constant, not a repeated literal (S1192).
_DEFAULT_PROFILE = "default"


def _adapt_tools(tools: list[dict[str, Any]] | None) -> tuple[str, ...] | dict[str, Any]:
    """Validate + adapt the MCP ``tools`` array to CLI ``--tool`` specs.

    Returns the spec tuple on success, or a structured error dict (to return to
    the agent) when an item is malformed — so a bad ``tools`` payload fails
    cleanly rather than as an uncaught error once generation is wired.
    """
    from pydantic import ValidationError

    from gflow_cli.tools.invocation import tool_specs_from_invocations

    try:
        return tool_specs_from_invocations(tools)
    except ValidationError as exc:
        log.warning("mcp.tool.invalid_tools", error=str(exc))
        return {
            "status": "invalid_tools",
            "error": (
                "Each item in 'tools' must be {'name': <slug>, 'options': {k: v}}. "
                f"Validation failed: {exc}"
            ),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_and_validate_profile(profile: str) -> str | dict[str, Any]:
    """Resolve the requested profile name using the same precedence as the CLI.

    When *profile* is ``"default"`` (the MCP sentinel meaning "auto-pick"),
    this runs the full CLI resolution chain:

    1. ``GFLOW_CLI_PROFILE`` env var
    2. ``config.toml`` ``default_profile``
    3. Auto-select the only profile that exists on disk

    When *profile* is any other value (an explicit name the agent passed in),
    it is forwarded to ``resolve_profile()`` as-is so the same validation runs.

    Returns the resolved profile name string on success, or a ready-to-return
    error dict on failure.
    """
    try:
        # Pass None when the agent omitted the profile (left it as "default")
        # so resolve_profile runs the full auto-detection chain.
        cli_flag: str | None = None if profile == "default" else profile
        resolved = resolve_profile(cli_flag)
    except NoProfilesError:
        return {
            "status": "error",
            "error": {
                "type": "https://gflow-cli.dev/errors/no-profile",
                "title": "No Profile Found",
                "status": 400,
                "detail": (
                    "No gflow profiles exist. Run `gflow auth login --browser chrome` first."
                ),
            },
        }
    except NoDefaultProfileError as exc:
        return {
            "status": "error",
            "error": {
                "type": "https://gflow-cli.dev/errors/no-default-profile",
                "title": "No Default Profile",
                "status": 400,
                "detail": (
                    f"Multiple profiles exist ({', '.join(exc.available)}) but none is set as "
                    "default. Pass profile=<name> explicitly, or run "
                    "`gflow auth use <name>` / set GFLOW_CLI_PROFILE."
                ),
                "available_profiles": exc.available,
            },
        }

    # Sanity-check: the profile directory must exist on disk. If auth was never
    # completed the FlowApiClient would fail with a cryptic Playwright error;
    # surface a clear message here instead.
    settings = get_settings()
    profile_dir = settings.profile_subdir(resolved)
    if not profile_dir.exists():
        return {
            "status": "error",
            "error": {
                "type": "https://gflow-cli.dev/errors/no-profile",
                "title": "Profile Directory Not Found",
                "status": 400,
                "detail": (
                    f"Profile {resolved!r} resolved but its directory does not exist: "
                    f"{profile_dir}. Run `gflow auth login --browser chrome` first."
                ),
            },
        }

    log.debug("mcp.tool.profile_resolved", requested=profile, resolved=resolved)
    return resolved


def _enqueue_generation_task(
    *,
    profile: str,
    task_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Enqueue a generation task into generation_queue and return task handle immediately."""
    settings = get_settings()
    db_path = settings.resolved_db_path()
    task_id = str(uuid.uuid4())

    with DataStore.open(db_path) as store:
        data_repo = DataRepository(store)
        profile_dir = settings.profile_subdir(profile)
        data_repo.upsert_profile(profile, profile_dir)

        ref_err = _resolve_payload_refs(data_repo, profile, payload, task_type)
        if ref_err is not None:
            return ref_err

        versioned_payload = dict(payload)
        versioned_payload.setdefault("schema_version", codec.CURRENT_SCHEMA_VERSION)

        QueueRepository(store).enqueue_task(
            task_id=task_id,
            profile_name=profile,
            task_type=task_type,
            payload=versioned_payload,
        )

    log.info("mcp.tool.task_enqueued", task_id=task_id, task_type=task_type)
    return {
        "status": "pending",
        "task_id": task_id,
        "task_type": task_type,
    }


async def _run_generation_task(
    *,
    profile: str,
    task_type: str,
    payload: dict[str, Any],
    wait: bool = True,
) -> dict[str, Any]:
    """Enqueue a generation task and either return handle immediately or wait for completion."""
    if not wait:
        return _enqueue_generation_task(
            profile=profile,
            task_type=task_type,
            payload=payload,
        )

    settings = get_settings()
    db_path = settings.resolved_db_path()

    task_id = str(uuid.uuid4())

    try:
        # 1. Enqueue the task (short-lived connection, closed before the worker
        #    opens its own — WAL allows a single writer at a time).
        with DataStore.open(db_path) as store:
            data_repo = DataRepository(store)

            # Ensure the profile FK exists before inserting the queue row.
            profile_dir = settings.profile_subdir(profile)
            data_repo.upsert_profile(profile, profile_dir)

            # Resolve remote-ref UUIDs to local file paths (video paths only) so
            # the attach reuses the proven local-upload path; an unresolvable
            # UUID fails fast here instead of timing out in the browser (#237).
            ref_err = _resolve_payload_refs(data_repo, profile, payload, task_type)
            if ref_err is not None:
                return ref_err

            # Task C2: stamp the current queue schema version onto every
            # freshly built payload at the enqueue site (additive, top-level —
            # setdefault so it never overwrites a version already present).
            # Encoding is centralized here; decoding is centralized at
            # claim/execution.
            versioned_payload = dict(payload)
            versioned_payload.setdefault("schema_version", codec.CURRENT_SCHEMA_VERSION)

            QueueRepository(store).enqueue_task(
                task_id=task_id,
                profile_name=profile,
                task_type=task_type,
                payload=versioned_payload,
            )

        log.info(
            "mcp.tool.task_enqueued",
            task_id=task_id,
            task_type=task_type,
            profile=profile,
        )

        # 2. Atomically claim the task we just enqueued, then run it. The claim
        #    (shared with the daemon poll loop) is what guarantees this row is
        #    executed at most once. A None claim = invalid payload failed at
        #    claim time (no browser) or already taken; the read-back reports it.
        worker = FlowWorker(profile_name=profile, db_path=str(db_path))
        try:
            claimed = worker.repo.claim_task(task_id, claimant=f"mcp:{profile}")
            if claimed is not None:
                await worker.process_task(claimed)
        finally:
            worker.close()

        # 3. Read the final task state back.
        with DataStore.open(db_path) as store:
            completed_task = QueueRepository(store).get_task(task_id)
            if completed_task is None:
                return {
                    "status": "error",
                    "error": f"Task {task_id!r} disappeared from queue after execution.",
                }

            # Treat anything other than an explicit "completed" as a failure —
            # a row stuck in "processing"/"pending" must not be reported as a
            # success with an empty file list.
            if completed_task.status != "completed":
                log.warning(
                    "mcp.tool.task_failed",
                    task_id=task_id,
                    status=completed_task.status,
                    error=completed_task.error,
                )
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": completed_task.error
                    or {"detail": f"Task ended in unexpected status {completed_task.status!r}"},
                }

            # Resolve local file paths from the asset catalog.
            file_paths: list[str] = []
            flow_project_id: str | None = None
            flow_workflow_id: str | None = None
            if completed_task.flow_media_id:
                asset = DataRepository(store).get_asset_by_flow_media_id(
                    profile,
                    completed_task.flow_media_id,
                )
                if asset:
                    flow_project_id = asset.flow_project_id
                    flow_workflow_id = asset.flow_workflow_id
                    if asset.local_files:
                        file_paths = [
                            str(lf.path) for lf in asset.local_files if lf.path is not None
                        ]

        log.info(
            "mcp.tool.task_completed",
            task_id=task_id,
            flow_project_id=flow_project_id,
            flow_media_id=completed_task.flow_media_id,
            file_count=len(file_paths),
        )

        return {
            "status": "completed",
            "task_id": task_id,
            "flow_project_id": flow_project_id,
            "flow_media_id": completed_task.flow_media_id,
            "flow_workflow_id": flow_workflow_id,
            "files": file_paths,
        }

    except GFlowError as exc:
        log.error("mcp.tool.gflow_error", task_id=task_id, error=str(exc))
        return {
            "status": "error",
            "task_id": task_id,
            "error": _gflow_error_dict(exc),
        }
    except Exception as exc:
        log.exception("mcp.tool.unexpected_error", task_id=task_id, exc_info=exc)
        return {
            "status": "error",
            "task_id": task_id,
            "error": {
                "type": "https://gflow-cli.dev/errors/unknown",
                "title": "Unexpected Error",
                "status": 500,
                "detail": str(exc),
            },
        }


_BAD_PARAM_TYPE = "https://gflow-cli.dev/errors/bad-parameter"

# UUID (Flow media id) vs on-disk path discriminator for image references.


def _bad_param(title: str, detail: str) -> dict[str, Any]:
    """Build the standard RFC 9457 bad-parameter (400) error envelope."""
    return {
        "status": "error",
        "error": {"type": _BAD_PARAM_TYPE, "title": title, "status": 400, "detail": detail},
    }


def _resolve_ref_local_path(
    data_repo: DataRepository,
    profile: str,
    ref_id: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a remote media-id reference to a local image file path.

    v0.25.0 (#237 fix): generated media do not appear in Flow's frame/reference
    picker search — Flow does not index generation prompts and generated assets
    carry no display name — so a UUID ref cannot be attached by a picker name
    search (that was the release-blocking bug). Instead it is attached by
    re-using its already-on-disk file through the proven local-upload path.

    Returns ``(local_path, None)`` or ``(None, error_envelope)``. Fail-fast
    cases: a UUID absent from the catalog → *Reference Not Found*; a catalogued
    asset with no local file on disk → *Reference Not On Disk*. (Automatic
    download-by-media-id for the on-disk-missing case is a planned follow-up;
    for now the caller re-generates or passes a local path.)
    """
    asset = data_repo.get_asset_by_any_id(profile, ref_id)
    if asset is None:
        return None, _bad_param(
            "Reference Not Found",
            f"'{ref_id}' was not found in your asset catalog for profile "
            f"{profile!r}. Generate the image first, or pass a local file path.",
        )
    for local_file in asset.local_files:
        # A usable frame is an on-disk local file (cloud-only rows carry a
        # storage_provider and no local path — they can't feed a file upload).
        if (
            local_file.storage_provider is None
            and local_file.path is not None
            and local_file.path.is_file()
        ):
            return str(local_file.path), None
    return None, _bad_param(
        "Reference Not On Disk",
        f"'{ref_id}' is in your catalog but has no local image file on disk to "
        "attach. Re-generate it, or pass a local file path for the frame.",
    )


# Video task types whose media-id refs are resolved to local file paths and
# attached via the local-upload path (#237 fix — see _resolve_ref_local_path):
# the video FRAME picker does not surface generated media, so there is no
# existing tile to select. Image task types DO surface generated media in the
# reference picker, so their refs are enriched (not rewritten) and attached by
# selecting the existing asset — see _enrich_image_refs.
_VIDEO_TASK_TYPES = frozenset({"t2v", "i2v", "r2v"})


def _enrich_image_refs(
    data_repo: DataRepository,
    profile: str,
    payload: dict[str, Any],
) -> None:
    """Annotate each media-id ref in an image ``payload`` with its Flow
    ``display_name`` and on-disk ``local_path`` (best-effort), so the transport
    can attach it by **selecting the already-existing Flow asset** in the
    reference picker — the preferred path (no duplicate upload) — with the local
    file as an upload fallback. Never errors: an uncatalogued UUID is still a
    valid media id to attach in place (PR #245 — image refs pass through).
    """
    refs = payload.get("refs")
    if not refs:
        return
    meta: dict[str, dict[str, str]] = {}
    for ref in refs:
        asset = data_repo.get_asset_by_any_id(profile, ref)
        if asset is None:
            continue
        entry = _ref_meta_entry(asset)
        if entry:
            meta[ref] = entry
    if meta:
        payload["ref_meta"] = meta


def _ref_meta_entry(asset: AssetLookup) -> dict[str, str]:
    """Build the ``display_name``/``local_path`` metadata entry for one catalog asset."""
    entry: dict[str, str] = {}
    name = asset.metadata_json.get("display_name")
    if isinstance(name, str) and name:
        entry["display_name"] = name
    for local_file in asset.local_files:
        if (
            local_file.storage_provider is None
            and local_file.path is not None
            and local_file.path.is_file()
        ):
            entry["local_path"] = str(local_file.path)
            break
    return entry


def _resolve_payload_refs(
    data_repo: DataRepository,
    profile: str,
    payload: dict[str, Any],
    task_type: str,
) -> dict[str, Any] | None:
    """Resolve media-id refs in ``payload`` in place, by task type.

    Video task types: a start/end frame ref becomes the local
    ``start_image``/``end_image`` path and r2v ``refs`` are merged into
    ``reference_images`` (local-upload attach — the video frame picker doesn't
    surface generated media). Image task types: refs are *enriched* with
    display_name/local_path so the transport selects the existing asset in the
    picker (see _enrich_image_refs). Returns an error envelope on the first
    unresolvable VIDEO UUID, else ``None`` (image enrichment never errors).
    """
    if task_type not in _VIDEO_TASK_TYPES:
        _enrich_image_refs(data_repo, profile, payload)
        return None
    if "refs" in payload:
        resolved_paths: list[str] = []
        for ref in payload["refs"]:
            path, err = _resolve_ref_local_path(data_repo, profile, ref)
            if err is not None:
                return err
            # path is never None when err is None (see _resolve_ref_local_path).
            assert path is not None
            resolved_paths.append(path)
        payload["reference_images"] = list(payload.get("reference_images", [])) + resolved_paths
        del payload["refs"]
    for ref_key, path_key in (("start_image_ref", "start_image"), ("end_image_ref", "end_image")):
        if ref_key in payload:
            path, err = _resolve_ref_local_path(data_repo, profile, payload[ref_key])
            if err is not None:
                return err
            payload[path_key] = path
            del payload[ref_key]
    return None


def _validate_project(project: str | None) -> dict[str, Any] | None:
    """Return a bad-parameter error dict if ``project`` is set but not a valid
    Flow project id, else ``None``. Reuses the CLI's ``_FLOW_ID_RE`` so the MCP
    ``project`` arg is validated identically to the CLI ``--project`` flag.
    """
    if project is not None and not _FLOW_ID_RE.fullmatch(project):
        return _bad_param(
            "Invalid Project Id",
            f"Project id '{project}' is not a valid Flow project id.",
        )
    return None


def _resolve_image_path(
    raw: str, *, title: str, label: str
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve a user-supplied image path.

    Returns ``(resolved_path, None)`` when ``raw`` is an existing file, or
    ``(None, error_dict)`` with an RFC 9457 bad-parameter error otherwise.
    Shared by the image and video tools so the validation message stays uniform.
    """
    path = Path(raw).resolve()
    if not path.is_file():
        return None, _bad_param(title, f"{label} '{raw}' does not exist or is not a file.")
    return str(path), None


def _resolve_image_references(
    reference_images: list[str],
) -> tuple[dict[str, list[str]] | None, dict[str, Any] | None]:
    """Split image ``reference_images`` into Flow-media-id refs vs resolved
    on-disk paths. Returns ``({"refs", "ref_paths"}, None)`` or ``(None, error)``.
    """
    refs: list[str] = []
    ref_paths: list[str] = []
    for ref in reference_images:
        if is_media_uuid(ref):
            refs.append(ref)
            continue
        resolved, err = _resolve_image_path(
            ref, title="Invalid Reference Image", label="Reference image path"
        )
        if err is not None:
            return None, err
        assert resolved is not None
        ref_paths.append(resolved)
    return {"refs": refs, "ref_paths": ref_paths}, None


def _build_video_media_inputs(
    *,
    mode: str,
    initial_frame: str | None,
    end_frame: str | None,
    reference_images: list[str] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate + resolve the media inputs (start/end/reference frames) for a
    video request. Returns ``(payload_fragment, None)`` or ``(None, error)``.

    Enforces mutual exclusivity (r2v refs vs i2v frames) and the mode-specific
    required inputs at the tool boundary, so a missing frame fails fast with a
    clear 400 instead of a cryptic worker ``ValueError``.
    """
    if reference_images and (initial_frame or end_frame):
        return None, _bad_param(
            "Mutually Exclusive Arguments",
            "reference_images (for r2v) cannot be used alongside initial_frame or "
            "end_frame (for i2v).",
        )
    if mode == "i2v" and initial_frame is None:
        return None, _bad_param(
            "Missing Start Image", "i2v (image-to-video) requires 'initial_frame'."
        )
    if mode == "r2v" and not reference_images:
        return None, _bad_param(
            "Missing Reference Images", "r2v (reference-to-video) requires 'reference_images'."
        )

    media: dict[str, Any] = {}
    for frame, ref_key, path_key, noun in (
        (initial_frame, "start_image_ref", "start_image", "Start"),
        (end_frame, "end_image_ref", "end_image", "End"),
    ):
        if frame is None:
            continue
        if is_media_uuid(frame):
            media[ref_key] = frame
            continue
        resolved, err = _resolve_image_path(
            frame, title=f"Invalid {noun} Image", label=f"{noun} image path"
        )
        if err is not None:
            return None, err
        media[path_key] = resolved
    if reference_images:
        ref_data, err = _resolve_image_references(reference_images)
        if err is not None:
            return None, err
        assert ref_data is not None
        if ref_data["ref_paths"]:
            media["reference_images"] = ref_data["ref_paths"]
        if ref_data["refs"]:
            media["refs"] = ref_data["refs"]
    return media, None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@server.tool(
    name="gflow_generate_image",
    description=(
        "Generate an image using Google Flow's Imagen model. "
        "Produces 1-4 images from a text prompt. "
        "Models: nano2 (fast), nano-pro (balanced), image4 (highest quality). "
        "Aspects: 1:1, 9:16, 16:9, 4:3, 3:4. "
        "The prompt supports @AssetName mentions to tag saved project characters/assets by name "
        "(resolves to referenceEntities/referenceImages). Reference a SAVED named asset via "
        "@Name; reference an arbitrary one-off image via reference_images. See "
        "docs/REFERENCE_STRATEGIES.md. "
        "Returns local file paths to the generated images."
    ),
)
async def gflow_generate_image(
    prompt: str,
    model: str = "nano2",
    aspect: str = "1:1",
    count: int = 1,
    seed: int | None = None,
    reference_images: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    profile: str = _DEFAULT_PROFILE,
    project: str | None = None,
    project_name: str | None = None,
    instructions: list[str] | None = None,
    ui_mode: str | None = None,
    output: str | None = None,
    wait: bool = True,
) -> dict[str, Any]:
    """Generate an image via Google Flow's Imagen.

    Args:
        prompt: The text prompt describing the desired image. Supports ``@AssetName``
            mentions to tag a saved project character or media asset by name (resolves to
            referenceEntities / referenceImages, deduped against reference_images). Use
            ``@Name`` for a saved named asset; use ``reference_images`` for an arbitrary
            one-off image. See ``docs/REFERENCE_STRATEGIES.md``.
        model: Model to use — 'nano2', 'nano-pro', or 'image4'.
        aspect: Aspect ratio — '1:1', '9:16', '16:9', '4:3', '3:4'.
        count: Number of images to generate (1-4).
        seed: Optional random seed for reproducibility.
        reference_images: Optional list of reference images for image-to-image generation.
            Can be local file paths or UUIDs of previously uploaded assets.
        tools: Optional list of prompt tools to apply before generation.
            Each item is ``{"name": str, "options": dict}``.  Valid names
            include ``"creative-director"`` (which supports an ``options``
            key of ``"style"`` for domain-vocabulary injection).
            Requires an OpenAI-compatible endpoint (GFLOW_CLI_LLM_API_KEY
            and/or GFLOW_CLI_LLM_BASE_URL); degrades gracefully to the
            original prompt when unavailable (mirrors the CLI ``--tool/-t``
            flag).
        profile: gflow-cli profile name to use.  Leave as ``"default"`` (or
            omit) to auto-resolve using the same precedence as the CLI:
            ``GFLOW_CLI_PROFILE`` env var → ``config.toml`` default →
            auto-select if exactly one profile exists.
        project: Optional existing Flow project id to generate into (mirrors the
            CLI ``--project`` flag). When omitted, a scratch project is created
            as before.
        project_name: Optional human-readable project title to use when creating a
            fresh Flow project.
        instructions: Optional list of custom agent instructions to add or enable
            (only in agentic mode).
        ui_mode: Required Flow UI arm — 'classic' (hard aspect controls),
            'agentic' (chat surface; forced automatically when instructions are
            given), or 'auto' (bind whatever renders, the default). If the arm
            can't be reached, generation aborts before submitting (no credits).
            Overrides GFLOW_CLI_UI_MODE.

    Returns:
        Dict with 'status', 'files' (list of local file paths), and metadata.
        On failure, 'status' is 'failed' or 'error' with an RFC 9457 'error' dict.
    """
    if (proj_err := _validate_project(project)) is not None:
        return proj_err

    if ui_mode is not None and ui_mode not in {m.value for m in UiMode}:
        return {
            "status": "error",
            "error": f"Invalid ui_mode {ui_mode!r}; expected one of {[m.value for m in UiMode]}.",
        }

    if not await _rate_limiter.acquire():
        log.warning("mcp.tool.rate_limited", tool="gflow_generate_image")
        return {
            "status": "rate_limited",
            "error": "Too many requests. Please wait before generating again.",
        }

    # Resolve and validate the profile BEFORE acquiring the per-profile lock so
    # that the lock key matches the real on-disk profile name, not the sentinel.
    resolved = _resolve_and_validate_profile(profile)
    if isinstance(resolved, dict):
        return resolved  # profile error — bail out early
    resolved_profile = resolved

    log.info(
        "mcp.tool.generate_image",
        prompt=prompt[:80],
        model=model,
        aspect=aspect,
        count=count,
        profile=resolved_profile,
    )

    # Validate + adapt the agent-supplied tools array to CLI --tool specs.
    adapted = _adapt_tools(tools)
    if isinstance(adapted, dict):
        return adapted
    tool_specs = adapted

    payload: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "aspect": aspect,
        "count": count,
    }
    if instructions:
        payload["instructions"] = list(instructions)
    if ui_mode is not None:
        payload["ui_mode"] = ui_mode
    if seed is not None:
        payload["seed"] = seed
    if project is not None:
        payload["project_id"] = project
    if project_name is not None:
        payload["project_name"] = project_name
    if output is not None:
        payload["output_file"] = output

    task_type = "t2i"
    if reference_images:
        ref_data, err = _resolve_image_references(reference_images)
        if err is not None:
            return err
        assert ref_data is not None
        payload["refs"] = ref_data["refs"]
        payload["ref_paths"] = ref_data["ref_paths"]
        task_type = "i2i"

    if tool_specs:
        payload["tool_specs"] = list(tool_specs)

    result = await _run_generation_task(
        profile=resolved_profile,
        task_type=task_type,
        payload=payload,
        wait=wait,
    )

    # Annotate the result with the original request parameters for context.
    result["params"] = {
        "prompt": prompt,
        "model": model,
        "aspect": aspect,
        "count": count,
        "seed": seed,
        "reference_images": reference_images,
        "tools": tools or [],
        "tool_specs": list(tool_specs),
        "profile": resolved_profile,
        "requested_profile": profile,
        "project": project,
    }
    return result


def _build_video_payload(
    prompt: str,
    mode: str,
    aspect: str,
    count: int,
    model: str | None,
    duration: int | None,
    tool_specs: Any,
    project: str | None,
    project_name: str | None,
    output: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "mode": mode,
        "aspect": aspect,
        "count": count,
    }
    if model is not None:
        payload["model"] = model
    if duration is not None:
        payload["duration"] = duration
    if tool_specs:
        payload["tool_specs"] = list(tool_specs)
    if project is not None:
        payload["project_id"] = project
    if project_name is not None:
        payload["project_name"] = project_name
    if output is not None:
        payload["output_file"] = output
    return payload


@server.tool(
    name="gflow_generate_video",
    description=(
        "Generate a video using Google Flow's Veo model. "
        "Modes: t2v (text-to-video), i2v (image-to-video), r2v (reference-to-video). "
        "Aspects: 9:16, 16:9. "
        "Optional model (veo_lite/veo_fast/veo_quality/omni_flash), duration (seconds), "
        "and count select the Veo model, clip length, and batch size (CLI parity). "
        "The prompt supports @CharacterName mentions to tag saved project characters by name "
        "(resolves to referenceEntities). Reference a SAVED character via @Name; pass one-off "
        "ingredient images via reference_images. See docs/REFERENCE_STRATEGIES.md. "
        "Returns the local file path to the generated video."
    ),
)
async def gflow_generate_video(  # NOSONAR
    prompt: str,
    mode: str = "t2v",
    aspect: str = "9:16",
    initial_frame: str | None = None,
    end_frame: str | None = None,
    reference_images: list[str] | None = None,
    model: str | None = None,
    duration: int | None = None,
    count: int = 1,
    tools: list[dict[str, Any]] | None = None,
    profile: str = _DEFAULT_PROFILE,
    project: str | None = None,
    project_name: str | None = None,
    output: str | None = None,
    wait: bool = True,
) -> dict[str, Any]:
    """Generate a video via Google Flow's Veo.

    Args:
        prompt: The text prompt describing the desired video. Supports ``@CharacterName``
            mentions to tag a saved project character by name (resolves to referenceEntities).
            Use ``@Name`` for a saved character; use ``reference_images`` for one-off ingredient
            images. See ``docs/REFERENCE_STRATEGIES.md``.
        mode: Generation mode — 't2v', 'i2v', or 'r2v'.
        aspect: Aspect ratio — '9:16' or '16:9'.
        initial_frame: Path to start frame image (required for i2v).
        end_frame: Path to end frame image (optional for i2v).
        reference_images: List of reference image paths (ingredients) for r2v.
        model: Optional Veo model — 'veo_lite', 'veo_fast', 'veo_quality',
            'omni_flash' (aliases accepted, mirrors the CLI ``--model``). When
            omitted, Flow's UI default applies EXCEPT for i2v with frames,
            where the transport defaults to veo-lite. Note: 'omni_flash'
            supports start-frame i2v (wire-verified 2026-08-03, refs #125) but
            NOT an end frame — i2v with 'end_frame' requires a Veo 3.1 model.
        duration: Optional clip length in seconds (mirrors the CLI ``--duration``).
            When omitted, Flow's per-model default applies.
        count: Number of videos to generate (mirrors the CLI ``--count``; default 1).
        tools: Optional list of prompt tools to apply before generation.
            Each item is ``{"name": str, "options": dict}``.  Valid names
            include ``"creative-director"`` (which supports an ``options``
            key of ``"style"`` for domain-vocabulary injection).
            Requires an OpenAI-compatible endpoint (GFLOW_CLI_LLM_API_KEY
            and/or GFLOW_CLI_LLM_BASE_URL); degrades gracefully to the
            original prompt when unavailable (mirrors the CLI ``--tool/-t``
            flag on ``video t2v``).
        profile: gflow-cli profile name to use.  Leave as ``"default"`` (or
            omit) to auto-resolve using the same precedence as the CLI:
            ``GFLOW_CLI_PROFILE`` env var → ``config.toml`` default →
            auto-select if exactly one profile exists.
        project: Optional existing Flow project id to generate into (mirrors the
            CLI ``--project`` flag on ``video t2v``/``i2v``/``r2v``). When
            omitted, a scratch project is created as before.
        **kwargs: Additional optional keyword arguments such as ``project_name``.

    Returns:
        Dict with 'status', 'files' (list of local file paths), and metadata.
        On failure, 'status' is 'failed' or 'error' with an RFC 9457 'error' dict.
    """
    if (proj_err := _validate_project(project)) is not None:
        return proj_err

    # Validate the model alias up front (mirrors the CLI's pre-spend check) so an
    # unknown model fails fast with a 400 instead of dying deep in the worker.
    if model is not None:
        from gflow_cli.api.video import VideoModel

        try:
            VideoModel.from_cli(model)
        except ValueError as exc:
            return _bad_param("Invalid Video Model", str(exc))

    if not await _rate_limiter.acquire():
        log.warning("mcp.tool.rate_limited", tool="gflow_generate_video")
        return {
            "status": "rate_limited",
            "error": {
                "type": "https://gflow-cli.dev/errors/rate-limited",
                "title": "Rate Limited",
                "status": 429,
                "detail": (
                    "Video generation rate limit reached (1 request per 30 seconds). "
                    "Please wait before making another request."
                ),
            },
        }

    # Resolve and validate the profile BEFORE acquiring the per-profile lock so
    # that the lock key matches the real on-disk profile name, not the sentinel.
    resolved_profile = _resolve_and_validate_profile(profile)
    if isinstance(resolved_profile, dict):
        return resolved_profile  # profile error — bail out early

    log.info(
        "mcp.tool.generate_video",
        prompt=prompt[:80],
        mode=mode,
        aspect=aspect,
        profile=resolved_profile,
    )

    adapted = _adapt_tools(tools)
    if isinstance(adapted, dict):
        return adapted
    tool_specs = adapted

    media, media_err = _build_video_media_inputs(
        mode=mode,
        initial_frame=initial_frame,
        end_frame=end_frame,
        reference_images=reference_images,
    )
    if media_err is not None:
        return media_err
    assert media is not None

    payload = _build_video_payload(
        prompt=prompt,
        mode=mode,
        aspect=aspect,
        count=count,
        model=model,
        duration=duration,
        tool_specs=tool_specs,
        project=project,
        project_name=project_name,
        output=output,
    )
    payload.update(media)

    # task_type matches the mode ("t2v", "i2v", "r2v")
    result = await _run_generation_task(
        profile=resolved_profile,
        task_type=mode,
        payload=payload,
        wait=wait,
    )

    # Annotate the result with the original request parameters for context.
    result["params"] = {
        "prompt": prompt,
        "mode": mode,
        "aspect": aspect,
        "initial_frame": initial_frame,
        "end_frame": end_frame,
        "reference_images": reference_images or [],
        "model": model,
        "duration": duration,
        "count": count,
        "tools": tools or [],
        "tool_specs": list(tool_specs),
        "profile": resolved_profile,
        "requested_profile": profile,
        "project": project,
        "project_name": project_name,
    }
    return result


def _normalize_voice_arg(value: str | None) -> str | dict[str, Any] | None:
    """Validate *value* against the voice catalog case-insensitively.

    Returns the canonical Capitalized voice name, ``None`` if *value* is
    ``None``, or a ready-to-return bad-parameter error dict if *value* doesn't
    match any known voice. Mirrors ``cli_character._normalize_voice`` — kept
    as its own small copy so ``mcp.tools`` doesn't reach into a CLI module for
    one helper.
    """
    if value is None:
        return None
    by_lower = {v.name.lower(): v.name for v in VOICES}
    canonical = by_lower.get(value.strip().lower())
    if canonical is None:
        return _bad_param(
            "Invalid Voice",
            f"Unknown voice {value!r}. Valid voices: {', '.join(v.name for v in VOICES)}.",
        )
    return canonical


@server.tool(
    name="gflow_create_character",
    description=(
        "Create a reusable Flow Character entity (face + optional body reference "
        "images, persisted server-side). Spends image-generation credits for each "
        "reference image. The character's name becomes referenceable via '@Name' "
        "mentions in later gflow_generate_image/gflow_generate_video prompts. "
        "Runs synchronously (no wait=False mode) since character creation is a "
        "short multi-step saga, not a long-poll queue task."
    ),
)
async def gflow_create_character(
    project: str,
    name: str,
    face_prompt: str,
    body_prompt: str | None = None,
    voice: str | None = None,
    personality: str | None = None,
    model: str = "nano2",
    format_prompt: bool = False,
    locale: str = "en-US",
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Create a Character entity in a Flow project.

    Args:
        project: Flow project id to create the character in (required — a
            Character is always project-scoped).
        name: Display name for the character. This is what downstream prompts
            reference via '@Name'.
        face_prompt: Prompt describing the character's face/headshot reference
            image (slot 0). Always generated.
        body_prompt: Optional prompt for a full-body reference image (slot 1).
            Omit for a face-only character.
        voice: Optional preset voice name (see the voice catalog); case-insensitive.
        personality: Optional personality/style brief attached to the entity.
        model: Image model alias for the reference generations (e.g. 'nano2').
        format_prompt: If True, runs the prompt through gflow's agent-mode
            prompt formatter before generating.
        locale: Locale tag forwarded to the creation flow.
        profile: gflow-cli profile name. 'default' auto-resolves like the CLI.

    Returns:
        Dict with 'status': 'ok' and a 'character' object (entity_id,
        project_id, name, workflow_ids, primary_media_ids, voice, image_paths),
        or 'status': 'error' with an RFC 9457 'error' dict.
    """
    if (proj_err := _validate_project(project)) is not None:
        return proj_err

    normalized_voice = _normalize_voice_arg(voice)
    if isinstance(normalized_voice, dict):
        return normalized_voice

    if not await _rate_limiter.acquire():
        log.warning("mcp.tool.rate_limited", tool="gflow_create_character")
        return {
            "status": "rate_limited",
            "error": "Too many requests. Please wait before generating again.",
        }

    resolved = _resolve_and_validate_profile(profile)
    if isinstance(resolved, dict):
        return resolved
    resolved_profile = resolved

    settings = get_settings()
    profile_dir = settings.profile_subdir(resolved_profile)
    log.info(
        "mcp.tool.create_character",
        project=project,
        name=name,
        model=model,
        profile=resolved_profile,
    )

    face = CharacterImageRequest(prompt=face_prompt, model=model, image_reference_index=0)
    body = (
        CharacterImageRequest(prompt=body_prompt, model=model, image_reference_index=1)
        if body_prompt is not None
        else None
    )

    try:
        async with FlowApiClient(profile_dir=profile_dir, headless=settings.headless) as client:
            recorder = OperationRecorder.open(settings)  # type: ignore[arg-type]
            try:
                result = await character_create(
                    client,
                    recorder,
                    profile_name=resolved_profile,
                    profile_dir=profile_dir,
                    project_id=project,
                    name=name,
                    face=face,
                    body=body,
                    voice=normalized_voice,
                    personality=personality,
                    locale=locale,
                    format_prompt=format_prompt,
                )
            finally:
                recorder.close()
    except GFlowError as exc:
        log.error("mcp.tool.create_character_gflow_error", error=str(exc))
        return _error_payload(_gflow_error_dict(exc))
    except Exception as exc:
        log.exception("mcp.tool.create_character_unexpected_error", exc_info=exc)
        return _error_payload(
            {
                "type": "https://gflow-cli.dev/errors/unknown",
                "title": "Unexpected Error",
                "status": 500,
                "detail": str(exc),
            }
        )

    return {
        "status": "ok",
        "character": {
            "entity_id": result.entity_id,
            "project_id": result.project_id,
            "name": result.name,
            "workflow_ids": list(result.workflow_ids),
            "primary_media_ids": list(result.primary_media_ids),
            "voice": result.voice,
            "image_paths": [str(p) if p is not None else None for p in result.image_paths],
        },
    }


# ---------------------------------------------------------------------------
# Codex-CLI-backed writer tools (Character Writer / Location Writer / Scenes)
# ---------------------------------------------------------------------------


async def _run_codex_writer(system_prompt: str, user_prompt: str) -> str | dict[str, Any]:
    """Shell out to the Codex CLI off the event loop; return text or an error dict."""
    from gflow_cli.tools.codex_cli import CodexCliError, run_codex_prompt

    try:
        return await asyncio.to_thread(run_codex_prompt, system_prompt, user_prompt)
    except CodexCliError as exc:
        log.error("mcp.tool.codex_cli_error", error=str(exc))
        return _error_payload(
            {
                "type": "https://gflow-cli.dev/errors/codex-cli",
                "title": "Codex CLI Error",
                "status": 502,
                "detail": str(exc),
            }
        )


@server.tool(
    name="gflow_write_character",
    description=(
        "Generate a character description from a brief via the Codex CLI. Backs "
        "the Character Writer node — its output feeds a Character Sheet node's "
        "'prompt' handle."
    ),
)
async def gflow_write_character(brief: str) -> dict[str, Any]:
    """Write a character description from *brief* using the Codex CLI.

    Args:
        brief: Free-form character brief (who they are, role, vibe).

    Returns:
        Dict with 'status': 'ok' and 'text', or 'status': 'error' with an
        RFC 9457 'error' dict.
    """
    from gflow_cli.tools.codex_cli import character_writer_prompt

    system, user = character_writer_prompt(brief)
    result = await _run_codex_writer(system, user)
    if isinstance(result, dict):
        return result
    return {"status": "ok", "text": result}


@server.tool(
    name="gflow_write_location",
    description=(
        "Generate a location description from a brief via the Codex CLI. Backs "
        "the Location Writer node — its output feeds a Location Sheet node's "
        "'prompt' handle."
    ),
)
async def gflow_write_location(brief: str) -> dict[str, Any]:
    """Write a location description from *brief* using the Codex CLI.

    Args:
        brief: Free-form location brief (setting, mood, era).

    Returns:
        Dict with 'status': 'ok' and 'text', or 'status': 'error' with an
        RFC 9457 'error' dict.
    """
    from gflow_cli.tools.codex_cli import location_writer_prompt

    system, user = location_writer_prompt(brief)
    result = await _run_codex_writer(system, user)
    if isinstance(result, dict):
        return result
    return {"status": "ok", "text": result}


@server.tool(
    name="gflow_write_scenes",
    description=(
        "Break a brief into a list of scenes via the Codex CLI. Backs the Scenes "
        "node — returns the scene list for display only; it does NOT create any "
        "nodes/edges. Use gflow_workflow_save separately to add a Text node per "
        "scene you want wired into an image/video node."
    ),
)
async def gflow_write_scenes(brief: str, scene_count: int | None = None) -> dict[str, Any]:
    """Break *brief* into a list of scenes using the Codex CLI.

    Args:
        brief: Free-form story/brief to break into scenes.
        scene_count: Optional exact scene count. Omit to let Codex decide.

    Returns:
        Dict with 'status': 'ok' and 'scenes' (list[str]), or 'status': 'error'
        with an RFC 9457 'error' dict.
    """
    from gflow_cli.tools.codex_cli import parse_scenes, scenes_prompt

    system, user = scenes_prompt(brief, scene_count)
    result = await _run_codex_writer(system, user)
    if isinstance(result, dict):
        return result
    return {"status": "ok", "scenes": parse_scenes(result)}


# ---------------------------------------------------------------------------
# chatgpt.com-backed sheet tools (Character Sheet / Location Sheet)
# ---------------------------------------------------------------------------


async def _generate_sheet(
    *,
    kind: str,
    prompt: str,
    model: str,
    aspect: str,
    count: int,
    profile: str,
) -> dict[str, Any]:
    """Shared body for gflow_generate_character_sheet/gflow_generate_location_sheet.

    Runs synchronously against a dedicated chatgpt.com Chrome profile (see
    ``chatgpt_profile_dir`` — never the Google Flow profile), same
    non-queued call shape as ``gflow_create_character``. ``model``/``aspect``
    are accepted for schema symmetry with ``gflow_generate_image`` and echoed
    back in ``params``, but chatgpt.com's image tool doesn't expose those as
    separate knobs — the only generation control is the prompt itself (which
    already carries the fixed 3-panel instruction, see
    ``chatgpt_ui_automation.build_sheet_prompt``).
    """
    from gflow_cli.api.transports.chatgpt_ui_automation import (
        ChatGptUiAutomationTransport,
        build_sheet_prompt,
        chatgpt_profile_dir,
    )

    if not prompt.strip():
        return _bad_param("Missing Prompt", "'prompt' must be non-empty.")

    if not await _rate_limiter.acquire():
        log.warning("mcp.tool.rate_limited", tool=f"gflow_generate_{kind}_sheet")
        return {
            "status": "rate_limited",
            "error": "Too many requests. Please wait before generating again.",
        }

    resolved = _resolve_and_validate_profile(profile)
    if isinstance(resolved, dict):
        return resolved
    resolved_profile = resolved

    settings = get_settings()
    profile_dir = chatgpt_profile_dir(resolved_profile)
    out_dir = settings.output_dir / f"{kind}_sheets" / str(uuid.uuid4())
    full_prompt = build_sheet_prompt(prompt)

    log.info(
        "mcp.tool.generate_sheet",
        kind=kind,
        profile=resolved_profile,
        model=model,
        aspect=aspect,
        count=count,
    )

    transport = ChatGptUiAutomationTransport()
    try:
        await transport.setup(profile_dir)
        files: list[str] = []
        for _ in range(max(1, count)):
            paths = await transport.generate_sheet_image(prompt=full_prompt, out_dir=out_dir)
            files.extend(str(p) for p in paths)
    except GFlowError as exc:
        log.error(f"mcp.tool.generate_{kind}_sheet_gflow_error", error=str(exc))
        return _error_payload(_gflow_error_dict(exc))
    except Exception as exc:
        log.exception(f"mcp.tool.generate_{kind}_sheet_unexpected_error", exc_info=exc)
        return _error_payload(
            {
                "type": "https://gflow-cli.dev/errors/unknown",
                "title": "Unexpected Error",
                "status": 500,
                "detail": str(exc),
            }
        )
    finally:
        await transport.teardown()

    if not files:
        return {
            "status": "error",
            "error": {
                "type": "https://gflow-cli.dev/errors/no-files",
                "title": "No Sheet Image Produced",
                "status": 502,
                "detail": (
                    "chatgpt.com automation completed without producing a "
                    "downloadable image — see chatgpt_ui_automation.py's "
                    "TODO(chatgpt-selectors)/TODO(chatgpt-network-filter) notes."
                ),
            },
        }

    return {
        "status": "completed",
        "files": files,
        "params": {"prompt": prompt, "model": model, "aspect": aspect, "count": count},
    }


@server.tool(
    name="gflow_generate_character_sheet",
    description=(
        "Generate a 3-panel character turnaround image (front/side/3-quarter view) "
        "from a character description, via a chatgpt.com browser-automation "
        "transport (real logged-in Chrome profile, network-response capture, "
        "download) mirroring gflow_generate_image's Google Flow transport. Backs "
        "the Character Sheet node."
    ),
)
async def gflow_generate_character_sheet(
    prompt: str,
    model: str = "nano2",
    aspect: str = "1:1",
    count: int = 1,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Generate a character sheet image from *prompt* via chatgpt.com.

    Args:
        prompt: Character description text (typically a Character Writer
            node's output) — the 3-panel instruction is added automatically.
        model: Accepted for schema symmetry with gflow_generate_image; echoed
            in 'params' but not applied (chatgpt.com exposes no model knob).
        aspect: Accepted for schema symmetry; echoed in 'params', not applied.
        count: Number of sheet images to generate sequentially.
        profile: gflow-cli profile name; resolves a *dedicated* chatgpt.com
            Chrome profile (see chatgpt_profile_dir), independent of the
            Google Flow profile of the same name.

    Returns:
        Dict with 'status': 'completed' and 'files' (local image paths), or
        'status': 'error'/'rate_limited' with details.
    """
    return await _generate_sheet(
        kind="character", prompt=prompt, model=model, aspect=aspect, count=count, profile=profile
    )


@server.tool(
    name="gflow_generate_location_sheet",
    description=(
        "Generate a 3-panel location turnaround image (front/side/3-quarter view) "
        "from a location description, via a chatgpt.com browser-automation "
        "transport (real logged-in Chrome profile, network-response capture, "
        "download) mirroring gflow_generate_image's Google Flow transport. Backs "
        "the Location Sheet node."
    ),
)
async def gflow_generate_location_sheet(
    prompt: str,
    model: str = "nano2",
    aspect: str = "1:1",
    count: int = 1,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Generate a location sheet image from *prompt* via chatgpt.com.

    Args:
        prompt: Location description text (typically a Location Writer
            node's output) — the 3-panel instruction is added automatically.
        model: Accepted for schema symmetry with gflow_generate_image; echoed
            in 'params' but not applied (chatgpt.com exposes no model knob).
        aspect: Accepted for schema symmetry; echoed in 'params', not applied.
        count: Number of sheet images to generate sequentially.
        profile: gflow-cli profile name; resolves a *dedicated* chatgpt.com
            Chrome profile (see chatgpt_profile_dir), independent of the
            Google Flow profile of the same name.

    Returns:
        Dict with 'status': 'completed' and 'files' (local image paths), or
        'status': 'error'/'rate_limited' with details.
    """
    return await _generate_sheet(
        kind="location", prompt=prompt, model=model, aspect=aspect, count=count, profile=profile
    )


def _resolve_workflow_project(project: str | None) -> str | dict[str, Any]:
    """Resolve *project* to a Flow project id, falling back to the GUI's last
    active project (``app_settings.json``'s ``activeProjectId``) when omitted.
    Returns a bad-parameter error dict if neither is available.
    """
    resolved = project or active_project_id()
    if resolved is None:
        return _bad_param(
            "Missing Project Id",
            "'project' was not given and no active gflow-director project was "
            "found (app_settings.json has no activeProjectId). Pass 'project' "
            "explicitly, or open gflow-director once so it binds a project.",
        )
    if (proj_err := _validate_project(resolved)) is not None:
        return proj_err
    return resolved


@server.tool(
    name="gflow_workflow_list",
    description=(
        "List gflow-director workflows (node-graph pipelines) saved for a project. "
        "Reads the same <workflow_id>.json files the gflow-director GUI reads/writes, "
        "so this sees workflows built in the app and vice versa."
    ),
)
async def gflow_workflow_list(project: str | None = None) -> dict[str, Any]:
    """List workflows for a project, newest-updated first.

    Args:
        project: Flow project id. Omit to use gflow-director's last active
            project (read from its app_settings.json).

    Returns:
        Dict with 'status', 'project', and 'workflows' (each: id, name, updatedAt).
    """
    resolved = _resolve_workflow_project(project)
    if isinstance(resolved, dict):
        return resolved

    summaries = list_workflows(resolved)
    return {
        "status": "ok",
        "project": resolved,
        "workflows": [{"id": s.id, "name": s.name, "updatedAt": s.updated_at} for s in summaries],
    }


@server.tool(
    name="gflow_workflow_get",
    description=(
        "Fetch a gflow-director workflow's full node graph (nodes + edges) by id. "
        "Node shapes: text ({text}), image ({model,aspect,count,jobId,artifactPath,...}), "
        "video ({model,aspect,duration,count,jobId,artifactPath,...}), character "
        "({name,facePrompt,bodyPrompt,voice,model,jobId,artifactPath,entityId,...}). "
        "Edges connect via sourceHandle/targetHandle: a Text node's 'text' handle "
        "feeds an Image/Video node's 'prompt' handle; an Image node's 'image' handle "
        "feeds a Video node's 'initialFrame' handle."
    ),
)
async def gflow_workflow_get(workflow_id: str, project: str | None = None) -> dict[str, Any]:
    """Fetch one workflow's full nodes/edges.

    Args:
        workflow_id: The workflow's id (from gflow_workflow_list or gflow_workflow_save).
        project: Flow project id. Omit to use gflow-director's last active project.

    Returns:
        Dict with 'status' and 'workflow' ({id, name, updatedAt, nodes, edges}),
        or 'status': 'error' if not found.
    """
    resolved = _resolve_workflow_project(project)
    if isinstance(resolved, dict):
        return resolved

    wf = get_workflow(resolved, workflow_id)
    if wf is None:
        return {
            "status": "error",
            "error": {
                "type": "https://gflow-cli.dev/errors/not-found",
                "title": "Workflow Not Found",
                "status": 404,
                "detail": f"No workflow {workflow_id!r} in project {resolved!r}.",
            },
        }
    return {"status": "ok", "project": resolved, "workflow": wf.to_json_dict()}


@server.tool(
    name="gflow_workflow_save",
    description=(
        "Create or overwrite a gflow-director workflow's node graph. Omit "
        "workflow_id to create a new workflow (a fresh id is generated); pass an "
        "existing workflow_id to overwrite it in place. Writes the same "
        "<workflow_id>.json file the GUI reads, so the result is immediately "
        "visible/editable there. See gflow_workflow_get's description for the "
        "node/edge shape."
    ),
)
async def gflow_workflow_save(
    name: str,
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
    workflow_id: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    """Create (workflow_id omitted) or overwrite (workflow_id given) a workflow.

    Args:
        name: Display name for the workflow.
        nodes: Full node list (see gflow_workflow_get for shapes). Omit/empty for
            a blank workflow.
        edges: Full edge list connecting node handles. Omit/empty for no connections.
        workflow_id: Existing workflow id to overwrite. Omitted = create new.
        project: Flow project id. Omit to use gflow-director's last active project.

    Returns:
        Dict with 'status': 'ok' and the saved 'workflow' ({id, name, updatedAt,
        nodes, edges}).
    """
    resolved = _resolve_workflow_project(project)
    if isinstance(resolved, dict):
        return resolved

    wid = workflow_id or str(uuid.uuid4())
    wf = save_workflow(resolved, wid, name, nodes or [], edges or [])
    log.info(
        "mcp.tool.workflow_save", project=resolved, workflow_id=wid, created=workflow_id is None
    )
    return {"status": "ok", "project": resolved, "workflow": wf.to_json_dict()}


def _node_error(node_id: str, node_type: str | None, detail: str) -> dict[str, Any]:
    return {"node_id": node_id, "node_type": node_type, "error": detail}


@server.tool(
    name="gflow_workflow_run",
    description=(
        "Run every generation node in a gflow-director workflow, in dependency "
        "order, resolving each node's inputs from its upstream edges (a Text "
        "node's text feeds a connected Image/Video node's prompt; an Image "
        "node's output feeds a connected Video node's initialFrame, making it "
        "i2v instead of t2v). Text nodes are inputs only (no generation runs "
        "for them). Character nodes run standalone (no edges). Writes each "
        "node's resulting jobId/artifactPath/entityId back into the workflow "
        "file, so gflow-director's GUI shows the same results. Always runs "
        "every node to completion before its dependents start — there is no "
        "fire-and-forget mode for a whole-graph run, since a downstream node "
        "needs its upstream node's actual output, not a pending handle."
    ),
)
async def gflow_workflow_run(
    workflow_id: str,
    project: str | None = None,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Run a workflow's Image/Video/Character nodes in dependency order.

    Args:
        workflow_id: The workflow to run (from gflow_workflow_list/save).
        project: Flow project id. Omit to use gflow-director's last active project.
        profile: gflow-cli profile name. 'default' auto-resolves like the CLI.

    Returns:
        Dict with 'status' ('ok' or 'partial_failure'), the updated 'workflow'
        (same shape as gflow_workflow_get, with results written into node data),
        'node_results' (per-node generation results, keyed by node id), and
        'failures' (list of {node_id, node_type, error} for any node that
        couldn't run — e.g. an Image node with no upstream Text prompt).
    """
    resolved = _resolve_workflow_project(project)
    if isinstance(resolved, dict):
        return resolved

    wf = get_workflow(resolved, workflow_id)
    if wf is None:
        return {
            "status": "error",
            "error": {
                "type": "https://gflow-cli.dev/errors/not-found",
                "title": "Workflow Not Found",
                "status": 404,
                "detail": f"No workflow {workflow_id!r} in project {resolved!r}.",
            },
        }

    nodes_by_id: dict[str, dict[str, Any]] = {n["id"]: n for n in wf.nodes}
    order = topological_order(wf.nodes, wf.edges)
    if order is None:
        return {
            "status": "error",
            "error": {
                "type": "https://gflow-cli.dev/errors/cycle-detected",
                "title": "Cycle Detected",
                "status": 400,
                "detail": "This workflow's prompt/initialFrame edges form a cycle.",
            },
        }

    log.info("mcp.tool.workflow_run", project=resolved, workflow_id=workflow_id, nodes=len(order))

    node_results: dict[str, Any] = {}
    node_updates: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for node_id in order:
        node = nodes_by_id[node_id]
        node_type = node.get("type")
        data = node.get("data", {})

        if node_type == "text":
            continue  # a Text node IS its data — nothing to generate

        if node_type == "character":
            face_prompt = data.get("facePrompt")
            if not face_prompt:
                failures.append(_node_error(node_id, node_type, "missing facePrompt"))
                continue
            result = await gflow_create_character(
                project=resolved,
                name=data.get("name") or "Character",
                face_prompt=face_prompt,
                body_prompt=data.get("bodyPrompt") or None,
                voice=data.get("voice") or None,
                model=data.get("model") or "nano2",
                profile=profile,
            )
            node_results[node_id] = result
            if result.get("status") == "ok":
                character = result["character"]
                image_paths: list[Any] = character.get("image_paths") or []
                node_updates[node_id] = {
                    "entityId": character.get("entity_id"),
                    "artifactPath": image_paths[0] if image_paths else None,
                }
            else:
                failures.append(_node_error(node_id, node_type, str(result.get("error"))))
            continue

        if node_type == "image":
            prompt = upstream_text(node_id, nodes_by_id, wf.edges)
            if not prompt:
                failures.append(
                    _node_error(node_id, node_type, "no upstream Text node wired to 'prompt'")
                )
                continue
            result = await gflow_generate_image(
                prompt=prompt,
                model=data.get("model") or "nano2",
                aspect=data.get("aspect") or "1:1",
                count=int(data.get("count") or 1),
                profile=profile,
                project=resolved,
            )
            node_results[node_id] = result
            if result.get("status") == "completed":
                files: list[Any] = result.get("files") or []
                node_updates[node_id] = {
                    "jobId": result.get("task_id"),
                    "artifactPath": files[0] if files else None,
                }
            else:
                failures.append(
                    _node_error(
                        node_id, node_type, str(result.get("error") or result.get("status"))
                    )
                )
            continue

        if node_type == "video":
            prompt = upstream_text(node_id, nodes_by_id, wf.edges)
            if not prompt:
                failures.append(
                    _node_error(node_id, node_type, "no upstream Text node wired to 'prompt'")
                )
                continue
            initial_frame = upstream_image_path(node_id, nodes_by_id, wf.edges)
            duration = data.get("duration")
            result = await gflow_generate_video(
                prompt=prompt,
                mode="i2v" if initial_frame else "t2v",
                aspect=data.get("aspect") or "16:9",
                initial_frame=initial_frame,
                model=data.get("model") or None,
                duration=int(duration) if duration is not None else None,
                count=int(data.get("count") or 1),
                profile=profile,
                project=resolved,
            )
            node_results[node_id] = result
            if result.get("status") == "completed":
                files: list[Any] = result.get("files") or []
                node_updates[node_id] = {
                    "jobId": result.get("task_id"),
                    "artifactPath": files[0] if files else None,
                }
            else:
                failures.append(
                    _node_error(
                        node_id, node_type, str(result.get("error") or result.get("status"))
                    )
                )
            continue

        if node_type in ("characterWriter", "locationWriter"):
            brief = data.get("brief")
            if not brief:
                failures.append(_node_error(node_id, node_type, "missing brief"))
                continue
            writer = gflow_write_character if node_type == "characterWriter" else gflow_write_location
            result = await writer(brief=brief)
            node_results[node_id] = result
            if result.get("status") == "ok":
                node_updates[node_id] = {"text": result.get("text")}
            else:
                failures.append(_node_error(node_id, node_type, str(result.get("error"))))
            continue

        if node_type == "scenes":
            brief = data.get("brief")
            if not brief:
                failures.append(_node_error(node_id, node_type, "missing brief"))
                continue
            scene_count = data.get("sceneCount")
            result = await gflow_write_scenes(
                brief=brief,
                scene_count=int(scene_count) if scene_count else None,
            )
            node_results[node_id] = result
            if result.get("status") == "ok":
                # Only the node's own data is patched — no new nodes/edges are
                # created here. Scene notes are added on demand from the UI
                # (Scenes node's per-scene "+ Add note" button), never
                # auto-generated by a run.
                node_updates[node_id] = {"scenes": result.get("scenes")}
            else:
                failures.append(_node_error(node_id, node_type, str(result.get("error"))))
            continue

        if node_type in ("characterSheet", "locationSheet"):
            prompt = upstream_text(node_id, nodes_by_id, wf.edges)
            if not prompt:
                failures.append(
                    _node_error(
                        node_id,
                        node_type,
                        "no upstream Text/Character-Writer/Location-Writer node wired to 'prompt'",
                    )
                )
                continue
            sheet_tool = (
                gflow_generate_character_sheet
                if node_type == "characterSheet"
                else gflow_generate_location_sheet
            )
            result = await sheet_tool(
                prompt=prompt,
                model=data.get("model") or "nano2",
                aspect=data.get("aspect") or "1:1",
                count=int(data.get("count") or 1),
                profile=profile,
            )
            node_results[node_id] = result
            if result.get("status") == "completed":
                files: list[Any] = result.get("files") or []
                node_updates[node_id] = {"artifactPath": files[0] if files else None}
            else:
                failures.append(
                    _node_error(
                        node_id, node_type, str(result.get("error") or result.get("status"))
                    )
                )
            continue

        failures.append(_node_error(node_id, node_type, f"unknown node type {node_type!r}"))

    patched_nodes = [
        {**node, "data": {**node.get("data", {}), **node_updates[node["id"]]}}
        if node["id"] in node_updates
        else node
        for node in wf.nodes
    ]
    saved = save_workflow(resolved, workflow_id, wf.name, patched_nodes, wf.edges)

    return {
        "status": "ok" if not failures else "partial_failure",
        "project": resolved,
        "workflow": saved.to_json_dict(),
        "node_results": node_results,
        "failures": failures,
    }


@server.tool(
    name="gflow_list_tools",
    description="List available gflow prompt tools (name, title, description, category).",
)
async def gflow_list_tools() -> dict[str, Any]:
    """List available prompt tools that can be passed to gflow_generate_image/video.

    Returns:
        Dict with 'tools' list; each entry has name, title, description, category.
    """
    from gflow_cli.tools.registry import iter_tools

    return {
        "tools": [
            {"name": s.name, "title": s.title, "description": s.description, "category": s.category}
            for s in iter_tools()
        ]
    }


@server.tool(
    name="gflow_list_projects",
    description=(
        "List all projects in the local gflow catalog. "
        "Returns project IDs, names, and creation dates from the SQLite database."
    ),
)
async def gflow_list_projects(
    profile: str = _DEFAULT_PROFILE,
    limit: int = 50,
) -> dict[str, Any]:
    """List projects from the local SQLite catalog.

    Args:
        profile: gflow-cli profile name to filter by.
        limit: Maximum number of projects to return.

    Returns:
        Dict with 'projects' list and pagination info.
    """
    log.info("mcp.tool.list_projects", profile=profile, limit=limit)

    settings = get_settings()
    db_path = settings.resolved_db_path()

    try:
        rows = list_projects(
            db_path=db_path,
            profile=profile if profile != "default" else None,
            limit=limit,
            offset=0,
        )
        return {
            "status": "ok",
            "projects": [
                {
                    "project_id": r.project_id,
                    "title": r.title,
                    "profile": r.profile,
                    "created_at": r.created_at.isoformat(),
                    "image_count": r.image_count,
                    "video_count": r.video_count,
                }
                for r in rows
            ],
            "total": len(rows),
        }
    except Exception as exc:
        log.error("mcp.tool.list_projects_error", error=str(exc))
        return {
            "status": "error",
            "error": str(exc),
            "projects": [],
            "total": 0,
        }


@server.tool(
    name="gflow_list_characters",
    description=(
        "List all Flow Character entities for the active profile. "
        "Characters are reusable project-scoped entities with voices."
    ),
)
async def gflow_list_characters(
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """List Flow Character entities from the local catalog.

    Characters are cloud-side Flow entities (not stored in the local SQLite
    catalog) and require an active browser session to enumerate.  Use
    ``gflow character list`` in the CLI for a full listing, or call this tool
    from a context where a browser session is available.

    Args:
        profile: gflow-cli profile name to filter by.

    Returns:
        Dict with 'characters' list.

    Note:
        Characters are project-scoped on the Flow side; this tool returns an
        empty list when no project_id is provided, since listing across all
        projects would require iterating every project.  Pass a project_id
        via a future parameter update, or use ``gflow character list`` in the
        terminal.
    """
    log.info("mcp.tool.list_characters", profile=profile)

    # Characters live on the Flow cloud side and are not cached in the local
    # SQLite catalog — fetching them requires an active browser session and a
    # specific project_id.  Return an informative empty response rather than
    # silently returning nothing or crashing.
    return {
        "status": "ok",
        "characters": [],
        "note": (
            "Character listing requires a project_id and an active browser session. "
            "Use `gflow character list --project <id>` in the terminal, or extend "
            "this MCP tool with a project_id parameter."
        ),
    }


# ---------------------------------------------------------------------------
# Instructions — persistent Agent-Mode brief cards (CLI parity: `gflow instructions`)
# ---------------------------------------------------------------------------
#
# These are credits-free brief PATCHes, not queued generations, so they open a
# FlowApiClient session directly (like the CLI `_run_*` helpers) instead of
# going through FlowWorker. All mutations are read-modify-write against the
# LIVE server brief with card ids preserved — same contract cli_instructions
# pins. No token-bucket: only generations burn credits; the per-profile lock
# still serialises browser sessions.


def _card_dict(card: AgentInstruction) -> dict[str, Any]:
    """One card in the stable JSON shape shared with `gflow instructions --json`."""
    return {
        "id": card.id,
        "title": card.resolved_title(),
        "enabled": card.enabled,
        "text": card.text,
        "image_media_ids": list(card.image_media_ids),
        "character_ids": list(card.character_ids),
    }


def _ok_payload(project: str, **extra: Any) -> dict[str, Any]:
    """Standard success envelope for the instructions tools."""
    return {"status": "ok", "project_id": project, **extra}


def _error_payload(error: dict[str, Any]) -> dict[str, Any]:
    """Standard failure envelope (mirrors the generate tools' shape)."""
    return {"status": "error", "error": error}


def _format_mcp_error(exc: GFlowError) -> str:
    """Format a GFlowError for MCP tool error responses.

    Includes the error class name, detail (or title if detail is empty),
    and remediation hint if present:
    `[error_class] detail (Remediation: remediation_hint)`
    """
    error_class = type(exc).__name__
    main_msg = exc.detail if exc.detail else exc.title
    if exc.remediation_hint:
        return f"[{error_class}] {main_msg} (Remediation: {exc.remediation_hint})"
    return f"[{error_class}] {main_msg}"


def _gflow_error_dict(exc: GFlowError) -> dict[str, Any]:
    """Problem-details dict + the shared retryable flag (``errors.is_retryable``)
    so the MCP envelope's retry signal stays identical to CLI ``--json`` and the
    worker queue — never fork a private retryable list here (§6.5)."""
    return {
        **exc.to_problem_details(),
        "message": _format_mcp_error(exc),
        "retryable": is_retryable(exc),
    }


def _selector_error(title: str | None, card_id: str | None) -> dict[str, Any] | None:
    """Enforce exactly one of title / card_id (mirrors the CLI selector rule)."""
    if (title is None) == (card_id is None):
        return _bad_param("Invalid Card Selector", "Provide exactly one of 'title' or 'card_id'.")
    return None


async def _run_instructions_op(
    *,
    tool: str,
    profile: str,
    project: str,
    op: Callable[[FlowApiClient], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Shared plumbing for the instructions tools.

    Validates the project id, resolves the profile, serialises on the
    per-profile lock, opens a FlowApiClient session, and maps failures to the
    standard envelopes: ``ValueError`` (card selection / card invariants) →
    RFC 9457 bad-parameter, ``GFlowError`` → its problem details, anything
    else → 500.
    """
    if not project:
        return _bad_param(
            "Missing Project Id",
            "'project' is required — persistent "
            "instruction cards only exist on a real Flow project.",
        )
    if (proj_err := _validate_project(project)) is not None:
        return proj_err

    resolved = _resolve_and_validate_profile(profile)
    if isinstance(resolved, dict):
        return resolved

    settings = get_settings()
    profile_dir = settings.profile_subdir(resolved)
    log.info("mcp.tool.instructions", tool=tool, project=project, profile=resolved)
    try:
        async with FlowApiClient(profile_dir=profile_dir, headless=settings.headless) as client:
            return await op(client)
    except ValueError as exc:
        # brief.find (not-found / ambiguous) and AgentInstruction invariants.
        return _bad_param("Invalid Instructions Request", str(exc))
    except GFlowError as exc:
        log.error("mcp.tool.instructions_gflow_error", tool=tool, error=str(exc))
        return _error_payload(_gflow_error_dict(exc))
    except Exception as exc:
        log.exception("mcp.tool.instructions_unexpected_error", tool=tool, exc_info=exc)
        return _error_payload(
            {
                "type": "https://gflow-cli.dev/errors/unknown",
                "title": "Unexpected Error",
                "status": 500,
                "detail": str(exc),
            }
        )


@server.tool(
    name="gflow_instructions_list",
    description=(
        "List a Flow project's persistent Agent-Mode instruction cards "
        "(reads the live server brief). Credits-free."
    ),
)
async def gflow_instructions_list(
    project: str,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """List the instruction cards on the project's brief.

    Args:
        project: Flow project id (required — briefs are project-scoped).
        profile: gflow-cli profile name; 'default' auto-resolves like the CLI.

    Returns:
        Dict with 'status', 'project_id', 'enabled' (brief master switch) and
        'cards' (id/title/enabled/text/image_media_ids/character_ids each).
    """

    async def _op(client: FlowApiClient) -> dict[str, Any]:
        brief = await client.get_agent_info(project)
        return _ok_payload(
            project,
            enabled=brief.enabled,
            cards=[_card_dict(c) for c in brief.cards],
        )

    return await _run_instructions_op(
        tool="gflow_instructions_list", profile=profile, project=project, op=_op
    )


@server.tool(
    name="gflow_instructions_add",
    description=(
        "Add a persistent instruction card to a Flow project's Agent-Mode brief "
        "(credits-free). Each ref is classified automatically: local image path "
        "→ uploaded as an image reference; asset UUID → image reference; "
        "anything else → character id/name."
    ),
)
async def gflow_instructions_add(
    project: str,
    title: str,
    text: str,
    refs: list[str] | None = None,
    enabled: bool = True,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Add an instruction card to the project's brief.

    Args:
        project: Flow project id (required).
        title: Human-readable card label (used for later selection by title).
        text: Guideline text the agent folds into every generation.
        refs: Optional references — local image paths, asset UUIDs, or
            character ids/names (classified automatically, like CLI --ref).
        enabled: Create the card enabled (default) or disabled.
        profile: gflow-cli profile name.

    Returns:
        Dict with 'status', 'project_id', and the created 'card'.
    """

    async def _op(client: FlowApiClient) -> dict[str, Any]:
        brief = await client.get_agent_info(project)
        image_ids, char_ids = await classify_refs(client, project, tuple(refs or ()))
        card = AgentInstruction(
            text=text,
            enabled=enabled,
            image_media_ids=tuple(image_ids),
            character_ids=tuple(char_ids),
            title=title,
        )
        # Send the FULL card set (existing + new): patch_agent_info REPLACES the
        # brief's cards, and existing cards keep their ids via read-modify-write.
        await client.patch_agent_info(project, enabled=True, cards=(*brief.cards, card))
        return _ok_payload(project, card=_card_dict(card))

    return await _run_instructions_op(
        tool="gflow_instructions_add", profile=profile, project=project, op=_op
    )


@server.tool(
    name="gflow_instructions_set_enabled",
    description=(
        "Enable or disable one instruction card on a Flow project's brief, "
        "selected by title or card id (exactly one). Credits-free."
    ),
)
async def gflow_instructions_set_enabled(
    project: str,
    enabled: bool,
    title: str | None = None,
    card_id: str | None = None,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Flip one card's enabled flag (covers CLI `instructions enable`/`disable`).

    Args:
        project: Flow project id (required).
        enabled: True to enable the card, False to disable it.
        title: Select the card by title (case-insensitive, must be unambiguous).
        card_id: Select the card by its stable server id.
        profile: gflow-cli profile name.

    Returns:
        Dict with 'status', 'project_id', and the updated 'card'.
    """
    if (sel_err := _selector_error(title, card_id)) is not None:
        return sel_err

    async def _op(client: FlowApiClient) -> dict[str, Any]:
        brief = await client.get_agent_info(project)
        card = brief.find(title=title) if card_id is None else brief.find(card_id=card_id)
        # Replace the target in place (preserving its id) and PATCH the FULL set.
        # cast: replace() loses the concrete type for Sonar S5655 (see sonar-dataclasses-replace).
        updated = cast("AgentInstruction", replace(card, enabled=enabled))  # pyright: ignore[reportUnnecessaryCast]
        new_cards = tuple(updated if c is card else c for c in brief.cards)
        await client.patch_agent_info(project, enabled=True, cards=new_cards)
        return _ok_payload(project, card=_card_dict(updated))

    return await _run_instructions_op(
        tool="gflow_instructions_set_enabled", profile=profile, project=project, op=_op
    )


@server.tool(
    name="gflow_instructions_rm",
    description=(
        "Remove one instruction card from a Flow project's brief, selected by "
        "title or card id (exactly one). Credits-free."
    ),
)
async def gflow_instructions_rm(
    project: str,
    title: str | None = None,
    card_id: str | None = None,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Remove the selected card from the project's brief.

    Args:
        project: Flow project id (required).
        title: Select the card by title (case-insensitive, must be unambiguous).
        card_id: Select the card by its stable server id.
        profile: gflow-cli profile name.

    Returns:
        Dict with 'status', 'project_id', and the removed 'card'.
    """
    if (sel_err := _selector_error(title, card_id)) is not None:
        return sel_err

    async def _op(client: FlowApiClient) -> dict[str, Any]:
        brief = await client.get_agent_info(project)
        card = brief.find(title=title) if card_id is None else brief.find(card_id=card_id)
        # Drop the target; send the remaining set (possibly empty -> clears it).
        new_cards = tuple(c for c in brief.cards if c is not card)
        await client.patch_agent_info(project, enabled=True, cards=new_cards)
        return _ok_payload(project, card=_card_dict(card))

    return await _run_instructions_op(
        tool="gflow_instructions_rm", profile=profile, project=project, op=_op
    )


@server.tool(
    name="gflow_instructions_toggle_mode",
    description=(
        "Turn a Flow project's brief master switch on or off. When off, NO "
        "cards apply even if individually enabled. Cards are left untouched. "
        "Credits-free."
    ),
)
async def gflow_instructions_toggle_mode(
    project: str,
    enabled: bool,
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Toggle the brief-level master switch (CLI `instructions toggle-mode`).

    Args:
        project: Flow project id (required).
        enabled: True for --on, False for --off.
        profile: gflow-cli profile name.

    Returns:
        Dict with 'status', 'project_id', and 'agent_mode_enabled'.
    """

    async def _op(client: FlowApiClient) -> dict[str, Any]:
        # Master switch only — cards are left untouched (no cards= mask sent).
        await client.patch_agent_info(project, enabled=enabled)
        return _ok_payload(project, agent_mode_enabled=enabled)

    return await _run_instructions_op(
        tool="gflow_instructions_toggle_mode", profile=profile, project=project, op=_op
    )


@server.tool(
    name="gflow_instructions_apply",
    description=(
        "Declaratively FULL-SYNC a Flow project's brief: REPLACES all existing "
        "instruction cards with the given set (destructive — cards not listed "
        "are removed). Each card is {'title', 'text', 'ref': [...], 'enabled'}. "
        "Credits-free."
    ),
)
async def gflow_instructions_apply(
    project: str,
    cards: list[dict[str, Any]],
    profile: str = _DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Full-sync the project's brief from a declarative card list.

    Args:
        project: Flow project id (required).
        cards: Declarative card entries; each is a dict with 'title' (str),
            'text' (str), optional 'ref' (list of image paths / asset UUIDs /
            character ids) and optional 'enabled' (bool, default True).
            This is the same entry shape as the CLI `instructions apply` file.
        profile: gflow-cli profile name.

    Returns:
        Dict with 'status', 'project_id', and the applied 'cards'.
    """

    async def _op(client: FlowApiClient) -> dict[str, Any]:
        built: list[AgentInstruction] = []
        for entry in cards:
            raw_refs = entry.get("ref", [])
            if not isinstance(raw_refs, list):
                msg = "each card's 'ref' must be a list of strings."
                raise ValueError(msg)
            refs = tuple(str(r) for r in cast("list[Any]", raw_refs))
            image_ids, char_ids = await classify_refs(client, project, refs)
            built.append(
                AgentInstruction(
                    text=str(entry.get("text", "")),
                    enabled=bool(entry.get("enabled", True)),
                    image_media_ids=tuple(image_ids),
                    character_ids=tuple(char_ids),
                    title=str(entry.get("title", "")),
                )
            )
        # Full replace: the given list is the declarative source of truth.
        await client.patch_agent_info(project, enabled=True, cards=tuple(built))
        return _ok_payload(project, cards=[_card_dict(c) for c in built])

    return await _run_instructions_op(
        tool="gflow_instructions_apply", profile=profile, project=project, op=_op
    )


# Re-export Path so tests that import it directly still work
__all__ = [
    "gflow_generate_image",
    "gflow_generate_video",
    "gflow_create_character",
    "gflow_workflow_list",
    "gflow_workflow_get",
    "gflow_workflow_save",
    "gflow_workflow_run",
    "gflow_list_tools",
    "gflow_list_projects",
    "gflow_list_characters",
    "gflow_instructions_list",
    "gflow_instructions_add",
    "gflow_instructions_set_enabled",
    "gflow_instructions_rm",
    "gflow_instructions_toggle_mode",
    "gflow_instructions_apply",
    "_TokenBucket",
    "_adapt_tools",
    "_format_mcp_error",
    "_gflow_error_dict",
    "_run_generation_task",
]
