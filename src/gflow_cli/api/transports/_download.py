# SPDX-License-Identifier: MIT
"""Shared "download generated media into out_dir" helper for browser-automation
transports.

Extracted from ``UiAutomationTransport._download``/``_is_allowed_download_host``
so a second transport (e.g. ``chatgpt_ui_automation.py``) can reuse the same
redirect-safety-sensitive download logic — host allowlisting plus
``follow_redirects=False`` — without duplicating it. ``UiAutomationTransport``
keeps its original staticmethod/function names as thin delegates to this
module so existing call sites and tests are unaffected.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import structlog

log = structlog.get_logger(__name__)


def is_allowed_download_host(url: str, allowed_host_suffixes: tuple[str, ...]) -> bool:
    """True if ``url``'s host ends with one of ``allowed_host_suffixes``.

    Refuses URLs that lack a host or use a non-https scheme — both shapes
    are unexpected for a legitimately-issued asset URL, and treating them as
    suspect is safer than treating them as trustworthy.
    """
    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return any(
        host == suffix or host.endswith("." + suffix) for suffix in allowed_host_suffixes
    )


async def download_images(
    urls: list[str],
    out_dir: Path,
    cookies: dict[str, str],
    *,
    allowed_host_suffixes: tuple[str, ...],
    log_prefix: str = "download",
) -> list[Path]:
    """Download each URL into ``out_dir`` using session cookies.

    Saves to ``out_dir / image_NN.<ext>`` (zero-padded index, jpg/png
    auto-detected from Content-Type / magic bytes). Individual download
    failures are logged and skipped — the function returns the list of paths
    that DID write successfully.

    URLs whose host is not in ``allowed_host_suffixes`` are skipped before
    any HTTP request is made — this prevents session cookies from being
    forwarded to an unexpected host through a malicious or compromised asset
    URL. Redirects are also disabled (``follow_redirects=False``) so an
    open-redirect on an allowed host cannot rebound the request to a third
    party.
    """
    import httpx  # local import — httpx is a runtime dependency

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=False,
        cookies=cookies,
    ) as client:
        for i, url in enumerate(urls):
            if not is_allowed_download_host(url, allowed_host_suffixes):
                log.error(
                    f"{log_prefix}.download_host_rejected",
                    url=url,
                    allowed_suffixes=list(allowed_host_suffixes),
                )
                continue
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                # Auto-detect extension from Content-Type / magic bytes.
                ct = resp.headers.get("content-type", "")
                if "jpeg" in ct or "jpg" in ct or resp.content[:3] == b"\xff\xd8\xff":
                    ext = ".jpg"
                else:
                    ext = ".png"
                p = out_dir / f"image_{i:02d}{ext}"
                p.write_bytes(resp.content)
                paths.append(p)
                log.info(
                    f"{log_prefix}.image_saved",
                    path=str(p),
                    bytes=len(resp.content),
                    format=ext,
                )
            except Exception as e:
                log.exception(
                    f"{log_prefix}.download_failed",
                    url=url,
                    error=str(e),
                )
    return paths
