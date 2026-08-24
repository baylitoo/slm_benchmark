"""Pins the Studio API's (method, path) route surface.

``studio_api.py`` was split from one 1000+ line module into a package
(``docie_bench/inngest/studio_api/``) for maintainability -- a pure move, no
route added/removed/renamed. A mis-wired ``include_router()`` (a submodule's
router built but never mounted onto the package's aggregate ``router``) fails
loudly at import time if a name is missing, but a silently-dropped router
object compiles fine and just serves 404 for that entire domain -- this test
is the safety net for that failure mode. Update ``EXPECTED_STUDIO_ROUTES``
deliberately whenever a route is genuinely added/removed/renamed (e.g.
``GET /v1/studio/seeds`` for the Downloads tab); the point is that every
change to this set is an intentional line in a diff, not a silent drop.

Uses ``app.openapi()`` rather than walking ``app.routes`` directly: FastAPI's
internal route-tree representation is not a flat list of ``Route`` objects in
every version (nested ``include_router`` calls produce internal wrapper
types), while the OpenAPI schema's ``paths`` mapping is the stable, public
way to enumerate the actual served (method, path) surface.
"""

from __future__ import annotations

import docie_bench.api as api

EXPECTED_STUDIO_ROUTES = {
    ("POST", "/v1/studio/extract"),
    ("POST", "/v1/studio/render-document"),
    ("GET", "/v1/studio/datasets"),
    ("POST", "/v1/studio/datasets/{name}/validate"),
    ("GET", "/v1/studio/model-profiles"),
    ("POST", "/v1/studio/model-profiles/pipeline"),
    ("POST", "/v1/studio/model-profiles/ocr"),
    ("DELETE", "/v1/studio/model-profiles/{name}"),
    ("GET", "/v1/studio/schemas/dynamic"),
    ("GET", "/v1/studio/schemas/dynamic/{name}"),
    ("POST", "/v1/studio/schemas/dynamic"),
    ("DELETE", "/v1/studio/schemas/dynamic/{name}"),
    ("GET", "/v1/studio/routing-policies"),
    ("GET", "/v1/studio/routing-policies/{name}"),
    ("POST", "/v1/studio/routing-policies"),
    ("DELETE", "/v1/studio/routing-policies/{name}"),
    ("POST", "/v1/studio/benchmark"),
    ("POST", "/v1/studio/deploy"),
    ("POST", "/v1/studio/seed-hf"),
    ("GET", "/v1/studio/hf/repo"),
    ("GET", "/v1/studio/hf/search"),
    ("GET", "/v1/studio/hf/inspect"),
    ("GET", "/v1/studio/hf/collection"),
    ("POST", "/v1/studio/seed-ollama"),
    ("GET", "/v1/studio/realtime-token"),
    ("GET", "/v1/studio/runs/{event_id}"),
    ("GET", "/v1/studio/runs"),
    ("POST", "/v1/studio/comparisons"),
    ("GET", "/v1/studio/artifacts/{artifact_id}"),
    ("GET", "/v1/studio/seeds"),
    # Batch extraction (N documents, one durable job, per-document state).
    ("POST", "/v1/studio/extract/batch"),
    ("GET", "/v1/studio/batches"),
    ("GET", "/v1/studio/batches/{event_id}"),
    ("GET", "/v1/studio/batches/{event_id}/results.{fmt}"),
}


def test_studio_api_route_surface_is_unchanged() -> None:
    schema = api.app.openapi()
    actual = {
        (method.upper(), path)
        for path, methods in schema["paths"].items()
        if path.startswith("/v1/studio")
        for method in methods
    }
    assert actual == EXPECTED_STUDIO_ROUTES
    assert len(EXPECTED_STUDIO_ROUTES) == 34
