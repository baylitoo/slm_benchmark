"""download_file resume: continue a partial .part across attempts (large-seed
retries) instead of restarting from zero."""

from __future__ import annotations

from pathlib import Path

import httpx

from docie_bench.serving.hf_hub import download_file

FULL = bytes(i % 251 for i in range(60_000))  # ~60 KB deterministic body


def _part_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".part")


def _ranged_handler(seen: list[str | None]):
    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("range")
        seen.append(rng)
        if rng:  # e.g. "bytes=20000-"
            start = int(rng.split("=", 1)[1].split("-", 1)[0])
            body = FULL[start:]
            return httpx.Response(
                206,
                content=body,
                headers={
                    "content-range": f"bytes {start}-{len(FULL) - 1}/{len(FULL)}",
                    "content-length": str(len(body)),
                },
            )
        return httpx.Response(200, content=FULL, headers={"content-length": str(len(FULL))})

    return handler


async def test_resumes_from_partial(tmp_path: Path) -> None:
    seen: list[str | None] = []
    dest = tmp_path / "model.gguf"
    _part_path(dest).write_bytes(FULL[:20_000])  # 20 KB already fetched

    async with httpx.AsyncClient(transport=httpx.MockTransport(_ranged_handler(seen))) as c:
        await download_file("owner/repo", "model.gguf", dest, client=c)

    assert dest.read_bytes() == FULL  # complete + correct (not duplicated)
    assert seen[0] == "bytes=20000-"  # asked to resume from the partial's size
    assert not _part_path(dest).exists()  # renamed to the canonical name


async def test_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    # Some backends answer 200 (full body) despite a Range header — must restart
    # from zero (truncate), never append onto the partial and duplicate bytes.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=FULL, headers={"content-length": str(len(FULL))})

    dest = tmp_path / "model.gguf"
    _part_path(dest).write_bytes(FULL[:20_000])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        await download_file("owner/repo", "model.gguf", dest, client=c)

    assert dest.read_bytes() == FULL  # exactly FULL — no 20 KB prefix duplicated


async def test_fresh_download_sends_no_range(tmp_path: Path) -> None:
    seen: list[str | None] = []
    dest = tmp_path / "model.gguf"  # no .part

    async with httpx.AsyncClient(transport=httpx.MockTransport(_ranged_handler(seen))) as c:
        await download_file("owner/repo", "model.gguf", dest, client=c)

    assert dest.read_bytes() == FULL
    assert seen == [None]  # no Range on a first download


async def test_skips_when_final_file_already_complete(tmp_path: Path) -> None:
    # A prior attempt finalized the file (canonical name, no .part). A re-run must
    # HEAD the remote, see the size matches, and skip — never re-GET the body.
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(FULL))})
        return httpx.Response(200, content=FULL, headers={"content-length": str(len(FULL))})

    dest = tmp_path / "model.gguf"
    dest.write_bytes(FULL)  # already complete, no .part

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        await download_file("owner/repo", "model.gguf", dest, client=c)

    assert dest.read_bytes() == FULL
    assert "GET" not in methods  # skipped the multi-GB re-download


async def test_redownloads_when_final_size_mismatches(tmp_path: Path) -> None:
    # A truncated/corrupt leftover at the canonical name (no .part) must NOT be
    # trusted: the HEAD size won't match, so it re-downloads to the full body.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(FULL))})
        return httpx.Response(200, content=FULL, headers={"content-length": str(len(FULL))})

    dest = tmp_path / "model.gguf"
    dest.write_bytes(FULL[:10_000])  # short leftover, no .part

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        await download_file("owner/repo", "model.gguf", dest, client=c)

    assert dest.read_bytes() == FULL


async def test_partial_keeps_part_on_network_error(tmp_path: Path) -> None:
    # A mid-download failure must KEEP the .part so the next attempt resumes it.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    dest = tmp_path / "model.gguf"
    part = _part_path(dest)
    part.write_bytes(FULL[:20_000])

    import contextlib

    from docie_bench.serving.hf_hub import HfHubError

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        with contextlib.suppress(HfHubError):
            await download_file("owner/repo", "model.gguf", dest, client=c)
    assert part.exists()  # partial retained for resume
    assert part.read_bytes() == FULL[:20_000]
