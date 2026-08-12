# SPDX-License-Identifier: MIT
"""Shell out to the OpenAI Codex CLI to generate text for writer/scene nodes.

TODO(codex-cli-invocation): this module assumes the installed ``codex`` CLI
supports a non-interactive one-shot mode invoked as
``codex exec --full-auto <prompt>`` (Codex CLI's documented automation mode).
If your installed version differs, adjust ``_CODEX_BIN``/``_CODEX_SUBCOMMAND``
via the ``GFLOW_CLI_CODEX_BIN``/``GFLOW_CLI_CODEX_SUBCOMMAND`` env vars, or
edit ``_build_argv`` below — everything else in this module (timeout
handling, error degrade, prompt builders) is independent of the exact CLI
flags.
"""

from __future__ import annotations

import os
import subprocess

import structlog

log = structlog.get_logger(__name__)

_CODEX_BIN = os.environ.get("GFLOW_CLI_CODEX_BIN", "codex")
_CODEX_SUBCOMMAND = os.environ.get("GFLOW_CLI_CODEX_SUBCOMMAND", "exec")
_DEFAULT_TIMEOUT_S = 120.0


class CodexCliError(Exception):
    """Raised when the codex CLI exits non-zero, times out, or produces no output."""


def _build_argv(system_prompt: str, user_prompt: str) -> list[str]:
    # Codex CLI's `exec` mode takes the whole instruction as a single
    # trailing positional argument; there's no separate system-prompt flag,
    # so the system instruction is prepended as a fenced preamble.
    combined = f"{system_prompt.strip()}\n\n---\n\n{user_prompt.strip()}"
    return [_CODEX_BIN, _CODEX_SUBCOMMAND, "--full-auto", combined]


def run_codex_prompt(
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> str:
    """Run one Codex CLI turn and return its final text output.

    Raises ``CodexCliError`` on non-zero exit, timeout, missing binary, or
    empty output — callers (the MCP tools in ``mcp/tools.py``) turn that into
    a ``{"status": "error", ...}`` payload rather than letting it propagate,
    matching this repo's degrade-don't-crash convention for external-process
    failures (see ``tools/runtime.py::_collect_frames``).
    """
    argv = _build_argv(system_prompt, user_prompt)
    log.info("codex_cli.run", bin=argv[0], subcommand=argv[1])  # never log full prompt text
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        log.warning("codex_cli.timeout", timeout_s=timeout_s)
        raise CodexCliError(f"codex CLI timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        log.warning("codex_cli.not_found", bin=_CODEX_BIN)
        raise CodexCliError(f"codex CLI not found on PATH ({_CODEX_BIN!r})") from exc

    if res.returncode != 0:
        log.warning("codex_cli.failed", code=res.returncode, stderr=res.stderr[-2000:])
        raise CodexCliError(f"codex CLI exited {res.returncode}: {res.stderr.strip()[:500]}")

    output = res.stdout.strip()
    if not output:
        raise CodexCliError("codex CLI produced no output")
    return output


def character_writer_prompt(brief: str) -> tuple[str, str]:
    """System/user prompt pair for the Character Writer node."""
    system = (
        "You are a character writer for a video-production pipeline. Given a "
        "brief, write a single cohesive character description covering: "
        "physical appearance, age, build, distinguishing features, wardrobe, "
        "and personality. Plain prose, no headings, no markdown, no "
        "preamble — return only the character description text."
    )
    return system, brief.strip()


def location_writer_prompt(brief: str) -> tuple[str, str]:
    """System/user prompt pair for the Location Writer node."""
    system = (
        "You are a location writer for a video-production pipeline. Given a "
        "brief, write a single cohesive location description covering: "
        "setting, time of day, lighting, atmosphere, and key visual "
        "landmarks. Plain prose, no headings, no markdown, no preamble — "
        "return only the location description text."
    )
    return system, brief.strip()


def scenes_prompt(brief: str, scene_count: int | None) -> tuple[str, str]:
    """System/user prompt pair for the Scenes node.

    Output is deliberately constrained to one scene per line (no numbering,
    no blank lines) so ``parse_scenes`` can split it back into a list
    without an intermediate JSON round-trip through the CLI.
    """
    count_instruction = f"exactly {scene_count}" if scene_count else "as many as the story needs"
    system = (
        f"You are a scene breakdown writer for a video-production pipeline. "
        f"Given a brief, break it into {count_instruction} scenes. Return "
        "ONLY the scenes, one per line, with no numbering, no headings, no "
        "markdown, and no blank lines between them — each line is one "
        "self-contained scene description usable directly as an image/video "
        "generation prompt."
    )
    return system, brief.strip()


def parse_scenes(output: str) -> list[str]:
    """Split Codex's one-scene-per-line output into a list of scene strings."""
    return [line.strip() for line in output.splitlines() if line.strip()]
