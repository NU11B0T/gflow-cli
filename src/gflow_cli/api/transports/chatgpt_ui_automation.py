# SPDX-License-Identifier: MIT
"""Browser-automation transport for chatgpt.com — backs the Character Sheet
and Location Sheet nodes.

Mirrors :class:`gflow_cli.api.transports.ui_automation.UiAutomationTransport`'s
shape: a Playwright-managed, persistent, real-Chrome-profile context against
a site with no public generation API, a network-response listener that
captures the generated asset's URL, and a download step through the shared
:mod:`gflow_cli.api.transports._download` helper. Scoped down for a first
version — no agentic/classic driver split, no video mixin, no request-body
capture; sheet generation is a single text-in/image-out call.

TWO THINGS ARE UNVERIFIED WITHOUT A LIVE BROWSER SESSION (flagged inline
with ``TODO(chatgpt-selectors)`` / ``TODO(chatgpt-network-filter)``):

1. The composer/submit selectors in ``_send_prompt`` — chatgpt.com's DOM
   structure. Verify with a live logged-in session before relying on this.
2. The response filter in ``_attach_image_response_listener`` — Google
   Flow's equivalent is a hardcoded ``batchGenerateImages`` URL substring;
   chatgpt.com's is unknown from this codebase, so the filter here is a
   broad heuristic (allowlisted asset host, or an ``image/*`` response) that
   should be tightened to an exact match once you've watched the real
   traffic in Chrome DevTools → Network, exactly as intended.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

import structlog

from gflow_cli.api.transports._download import download_images
from gflow_cli.profile_lease import ProfileLease

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from playwright.async_api import Page, ViewportSize

try:  # pragma: no cover — re-bound at module import in production
    from playwright.async_api import async_playwright
except ImportError:  # pragma: no cover — Playwright is an install dependency
    async_playwright = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)

CHATGPT_URL = "https://chatgpt.com/"

# Same real-desktop viewport as the Google Flow transport (ui_automation.py)
# — keeps the fingerprint consistent for anyone watching both sessions.
_VIEWPORT = {"width": 1920, "height": 1080}

# Hosts allowed when downloading a generated sheet image, plus the broad
# response-filter heuristic below. `oaiusercontent.com` is OpenAI's asset CDN
# for user-generated content as of this writing — confirm/adjust against the
# real traffic (see module docstring, item 2).
_ALLOWED_DOWNLOAD_HOST_SUFFIXES: tuple[str, ...] = (
    "oaiusercontent.com",
    "openai.com",
    "chatgpt.com",
)

# The fixed 3-panel-turnaround instruction appended to every sheet prompt —
# keeps CharacterSheetNode/LocationSheetNode's UI as plain as ImageNode's
# (model/aspect/count only); panel composition is not a user-facing field.
SHEET_PANEL_INSTRUCTION = (
    "Generate a single image containing exactly three labeled panels side by "
    "side on a plain neutral background: front view, side view, and 3-quarter "
    "view of the following subject. Keep proportions, colors, and details "
    "consistent across all three panels.\n\nSubject: "
)


def build_sheet_prompt(subject_text: str) -> str:
    return f"{SHEET_PANEL_INSTRUCTION}{subject_text.strip()}"


class ChatGptUiAutomationTransport:
    """Drives chatgpt.com's image tool through a logged-in persistent Chrome
    profile to produce one 3-panel turnaround image per call.

    Lifecycle::

        transport = ChatGptUiAutomationTransport()
        await transport.setup(profile_dir)
        paths = await transport.generate_sheet_image(prompt=..., out_dir=...)
        await transport.teardown()
    """

    name = "chatgpt_ui_automation"

    def __init__(self) -> None:
        self._pw_cm: Any | None = None
        self._ctx: Any | None = None
        self._page: Page | None = None
        self._setup_done: bool = False
        self._lease: ProfileLease | None = None
        # Serializes concurrent generate_sheet_image calls — one Page cannot
        # safely handle two concurrent prompt submissions (mirrors
        # UiAutomationTransport._generate_lock).
        self._generate_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self, profile_dir: Path) -> None:
        """Launch (or reuse) a persistent Chrome context logged into chatgpt.com.

        Idempotent — second call is a no-op. ``profile_dir`` should be a
        directory dedicated to this transport (never the Google Flow
        profile dir) — see ``chatgpt_profile_dir()`` below — so logging into
        chatgpt.com once here never touches the Flow session.
        """
        if self._setup_done:
            return

        from gflow_cli.api._engine import (
            CONTEXT_TEARDOWN_TIMEOUT_S,
            DRIVER_STOP_TIMEOUT_S,
            active_engine,
            close_context_bounded,
            log_engine_selected,
            resolve_async_playwright,
            run_teardown_step,
        )
        from gflow_cli.browser_manager import channel_for_profile
        from gflow_cli.config import BrowserEngine

        engine = active_engine()
        log_engine_selected(engine)
        if engine == BrowserEngine.PATCHRIGHT:
            pw_factory = resolve_async_playwright(engine)
        elif async_playwright is None:  # pragma: no cover — install-time guard
            msg = (
                "Playwright is required for ChatGptUiAutomationTransport. "
                "Install via `uv sync` (it is a runtime dependency)."
            )
            raise RuntimeError(msg)
        else:
            pw_factory = async_playwright

        pw_cm = pw_factory()
        pw: Any = await pw_cm.__aenter__()
        ctx = None
        try:
            import os

            # Own the profile BEFORE Chrome launches — same discipline as
            # UiAutomationTransport (contention raises ProfileLockedError
            # here; the except below tears the driver back down).
            self._lease = ProfileLease(profile_dir).acquire()
            locale_env = os.getenv("GFLOW_CLI_LOCALE", "en-US")
            ctx = await pw.chromium.launch_persistent_context(
                str(profile_dir),
                headless=False,
                viewport=cast("ViewportSize", _VIEWPORT),
                locale=locale_env,
                channel=channel_for_profile(profile_dir),
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--password-store=basic",
                    "--disable-dev-shm-usage",
                ],
            )
            await ctx.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})",
            )
            self._pw_cm = pw_cm
            self._ctx = ctx
            page = cast("Page", ctx.pages[0] if ctx.pages else await ctx.new_page())
            self._page = page
            try:
                await page.goto(CHATGPT_URL, wait_until="networkidle", timeout=45_000)
            except Exception as e:
                log.warning("chatgpt_ui_automation.initial_goto_failed", error=str(e))
            self._setup_done = True
            log.info("chatgpt_ui_automation.setup_own_context", profile_dir=str(profile_dir))
        except BaseException:
            # Partial-setup leak guard — mirrors UiAutomationTransport.setup's
            # BaseException handling so a cancelled launch cannot orphan
            # Chrome or leak the profile lease.
            if ctx is not None:
                await run_teardown_step(
                    close_context_bounded(ctx, owner="chatgpt_ui_automation"),
                    timeout=CONTEXT_TEARDOWN_TIMEOUT_S,
                    owner="chatgpt_ui_automation",
                    step="setup_context_close",
                )
            await run_teardown_step(
                pw_cm.__aexit__(None, None, None),
                timeout=DRIVER_STOP_TIMEOUT_S,
                owner="chatgpt_ui_automation",
                step="setup_driver_exit",
            )
            if self._lease is not None:
                self._lease.release()
                self._lease = None
            raise

    async def teardown(self) -> None:
        """Close the context and release the profile lease. Idempotent."""
        if not self._setup_done:
            return
        from gflow_cli.api._engine import (
            CONTEXT_TEARDOWN_TIMEOUT_S,
            DRIVER_STOP_TIMEOUT_S,
            close_context_bounded,
            run_teardown_step,
        )

        # Cancellation-complete teardown (mirrors UiAutomationTransport.teardown):
        # each step is bounded + shielded so a CancelledError landing mid-close
        # cannot skip the driver exit or the lease release in `finally`; the
        # original cancellation re-raises last.
        cancelled: BaseException | None = None
        try:
            if self._ctx is not None:
                cancelled = (
                    await run_teardown_step(
                        close_context_bounded(self._ctx, owner="chatgpt_ui_automation"),
                        timeout=CONTEXT_TEARDOWN_TIMEOUT_S,
                        owner="chatgpt_ui_automation",
                        step="teardown_context_close",
                    )
                    or cancelled
                )
            if self._pw_cm is not None:
                cancelled = (
                    await run_teardown_step(
                        self._pw_cm.__aexit__(None, None, None),
                        timeout=DRIVER_STOP_TIMEOUT_S,
                        owner="chatgpt_ui_automation",
                        step="teardown_driver_exit",
                    )
                    or cancelled
                )
        finally:
            if self._lease is not None:
                self._lease.release()
                self._lease = None
            self._pw_cm = None
            self._ctx = None
            self._page = None
            self._setup_done = False
        if cancelled is not None:
            raise cancelled

    # ------------------------------------------------------------------
    # Composer interaction
    # ------------------------------------------------------------------

    async def _send_prompt(self, page: Page, prompt_text: str) -> None:
        """Type ``prompt_text`` into ChatGPT's composer and submit it.

        TODO(chatgpt-selectors): ``#prompt-textarea`` (contenteditable) and
        ``[data-testid='send-button']`` are ChatGPT's commonly-documented
        structural hooks at time of writing — verify against the live DOM
        before depending on this in production, per this repo's rule
        (AGENTS.md) that transport selectors must be structural/ARIA, not
        text-label matches, and must be confirmed against the real app.
        """
        composer = page.locator("#prompt-textarea")
        await composer.click()
        # `.fill()` targets input/textarea value assignment; ChatGPT's composer
        # is a contenteditable div, so type via the keyboard instead — mirrors
        # how a real user's input event stream reaches the composer.
        await page.keyboard.type(prompt_text)
        send_button = page.locator("[data-testid='send-button']")
        await send_button.click()

    # ------------------------------------------------------------------
    # Network capture
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_image_response_listener(
        page: Page,
    ) -> tuple[list[dict[str, Any]], Callable[[], None]]:
        """Register a ``page.on('response', ...)`` listener that records
        candidate generated-image responses into a shared list.

        TODO(chatgpt-network-filter): unlike Google Flow's hardcoded
        ``batchGenerateImages`` substring match, this filter is a broad
        heuristic — any response whose host is in
        ``_ALLOWED_DOWNLOAD_HOST_SUFFIXES``, or whose content-type starts
        with ``image/``. Once you've watched the real traffic in DevTools →
        Network while submitting a prompt, replace this with an exact URL
        substring match (mirrors how ``batchGenerateImages`` was pinned for
        Flow).

        Returns ``(captured, detach_fn)`` — same contract as
        ``UiAutomationTransport._attach_batch_response_listener``.
        """
        captured: list[dict[str, Any]] = []

        async def on_response(response: Any) -> None:
            from urllib.parse import urlparse

            host = (urlparse(response.url).hostname or "").lower()
            host_allowed = any(
                host == suffix or host.endswith("." + suffix)
                for suffix in _ALLOWED_DOWNLOAD_HOST_SUFFIXES
            )
            content_type = ""
            try:
                headers = response.headers
                if callable(headers):
                    headers = await headers()
                content_type = (headers or {}).get("content-type", "")
            except Exception:
                pass
            is_image_response = content_type.startswith("image/")

            if not (host_allowed or is_image_response):
                return

            log.info(
                "chatgpt_ui_automation.candidate_response_seen",
                url=response.url,
                status=response.status,
                content_type=content_type,
            )
            captured.append(
                {
                    "url": response.url,
                    "status": response.status,
                    "content_type": content_type,
                    "ts": time.monotonic(),
                },
            )

        page.on("response", on_response)

        _detached = False

        def detach() -> None:
            nonlocal _detached
            if _detached:
                return
            _detached = True
            try:
                page.remove_listener("response", on_response)
            except Exception:
                pass

        return captured, detach

    @staticmethod
    async def _await_captured(
        captured: list[dict[str, Any]],
        timeout_s: float,
        *,
        submit_time: float,
        poll_interval_s: float = 0.5,
        straggler_window_s: float = 2.5,
    ) -> list[dict[str, Any]]:
        """Poll ``captured`` until at least one fresh (post-``submit_time``)
        entry arrives, then wait a short straggler window for the rest.
        Mirrors ``UiAutomationTransport._await_captured``'s shape (single-
        image expectation, since a sheet is always one image)."""
        deadline = time.monotonic() + timeout_s

        def _fresh() -> list[dict[str, Any]]:
            return [e for e in captured if e.get("ts", 0.0) >= submit_time]

        while time.monotonic() < deadline and not _fresh():
            await asyncio.sleep(poll_interval_s)

        fresh = _fresh()
        if not fresh:
            msg = f"No candidate image response within {timeout_s:.1f}s."
            raise TimeoutError(msg)

        await asyncio.sleep(straggler_window_s)
        return _fresh()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    async def generate_sheet_image(
        self,
        *,
        prompt: str,
        out_dir: Path,
        timeout_s: float = 180.0,
    ) -> list[Path]:
        """Submit ``prompt`` (already wrapped in the 3-panel instruction by
        the caller via ``build_sheet_prompt``) and return the downloaded
        image path(s).

        Serialized via ``self._generate_lock`` — one Page, one in-flight
        generation at a time, same discipline as
        ``UiAutomationTransport.generate_images``.
        """
        if not self._setup_done or self._page is None:
            msg = "ChatGptUiAutomationTransport.setup() must run before generate_sheet_image()"
            raise RuntimeError(msg)

        async with self._generate_lock:
            page = self._page
            captured, detach = self._attach_image_response_listener(page)
            try:
                submit_time = time.monotonic()
                await self._send_prompt(page, prompt)
                responses = await self._await_captured(
                    captured,
                    timeout_s,
                    submit_time=submit_time,
                )
            finally:
                detach()

            urls = [r["url"] for r in responses]
            cookies_list = await self._ctx.cookies()
            cookies = {c["name"]: c["value"] for c in cookies_list}
            return await download_images(
                urls,
                out_dir,
                cookies,
                allowed_host_suffixes=_ALLOWED_DOWNLOAD_HOST_SUFFIXES,
                log_prefix="chatgpt_ui_automation",
            )


def chatgpt_profile_dir(profile_name: str) -> Path:
    """Dedicated, Flow-profile-independent Chrome profile directory for the
    chatgpt.com session — sibling of the normal ``profile_<name>`` dir under
    the same gflow-cli home, e.g. ``chatgpt_profile_<name>``. Log into
    chatgpt.com once inside this profile (same one-time-login pattern used
    for the Flow profile) before running Character/Location Sheet nodes.
    """
    from gflow_cli.config import get_settings

    return get_settings().home / f"chatgpt_profile_{profile_name}"
