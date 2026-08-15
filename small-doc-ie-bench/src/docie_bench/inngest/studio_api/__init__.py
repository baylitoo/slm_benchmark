"""FastAPI glue for DocIE Studio: trigger jobs + consume results.

Mounted on the main API (``app.include_router(studio_router)``). Three routes
give the frontend everything it needs:

  - ``POST /v1/studio/extract``        -- fire ``doc/extract.requested``; returns
                                          the event id(s) and the realtime channel.
  - ``GET  /v1/studio/realtime-token`` -- mint a subscription token (the "hook").
  - ``GET  /v1/studio/runs/{id}``      -- polling fallback: proxy the Inngest
                                          server's run status for an event.

Realtime is best-effort; if the experimental API is unavailable the token route
returns 501 and the frontend falls back to polling ``/runs``.

Split from one ~1000-line module into this package for maintainability -- pure
move, no route added/removed/renamed (pinned by
tests/test_studio_api_route_inventory.py). One submodule per route domain:

  - ``extract.py``           -- extraction trigger + document rasterization
  - ``datasets.py``          -- registered-dataset listing/validation
  - ``model_profiles.py``    -- configs/models.yaml profile CRUD
  - ``dynamic_schemas.py``   -- named DynamicSchemaSpec CRUD
  - ``routing_policies.py``  -- named RoutingPolicy CRUD
  - ``benchmark.py``         -- benchmark trigger
  - ``deploy.py``            -- deploy/seed triggers + HF Hub search/inspect
  - ``runs.py``               -- realtime token, run status/listing, comparisons,
                                 artifact download

``_shared.py`` holds the constants/models/helpers every submodule needs
(``default_run_store``, ``MODELS_CONFIG_PATH``, ``_record_event_owners``,
etc.) -- see its own docstring for why those two names in particular are
accessed via ``_shared.<name>`` everywhere rather than ``from ._shared import
<name>``: it's what keeps a single ``monkeypatch.setattr(studio_api._shared,
...)`` effective for every route that touches them, regardless of which
submodule the route now lives in.

Every name that lived at this module's top level before the split (route
functions, request/response models, and the handful of underscore-prefixed
helpers a few tests import directly, e.g. ``_annotate_fits``) is re-exported
below so nothing outside this package needs to know it was ever one file.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import _shared
from ._shared import (
    _DOMAIN_404,
    BENCHMARK_EVENT,
    DEFAULT_TOPICS,
    DEPLOY_EVENT,
    EXTRACT_EVENT,
    MODELS_CONFIG_PATH,
    SEED_EVENT,
    TriggerResponse,
    _annotate_fits,
    _record_event_owners,
)
from .benchmark import BenchmarkRequest, trigger_benchmark
from .benchmark import router as _benchmark_router
from .datasets import (
    list_datasets,
    validate_dataset_version,
)
from .datasets import (
    router as _datasets_router,
)
from .deploy import (
    DeployRequest,
    SeedHfRequest,
    SeedOllamaRequest,
    _node_available_bytes,
    hf_collection,
    hf_inspect,
    hf_repo_ggufs,
    hf_search,
    trigger_deploy,
    trigger_seed_hf,
    trigger_seed_ollama,
)
from .deploy import (
    router as _deploy_router,
)
from .dynamic_schemas import (
    create_dynamic_schema_route,
    delete_dynamic_schema_route,
    get_dynamic_schema_route,
    list_dynamic_schemas_route,
)
from .dynamic_schemas import (
    router as _dynamic_schemas_router,
)
from .extract import (
    ExtractRequest,
    RenderDocumentRequest,
    render_document,
    trigger_extract,
)
from .extract import (
    router as _extract_router,
)
from .model_profiles import (
    OcrProfileRequest,
    PipelineProfileRequest,
    create_ocr_profile,
    create_pipeline_profile,
    delete_pipeline_profile_route,
    list_model_profiles,
)
from .model_profiles import (
    router as _model_profiles_router,
)
from .routing_policies import (
    RoutingPolicyCreateRequest,
    create_routing_policy_route,
    delete_routing_policy_route,
    get_routing_policy_route,
    list_routing_policies_route,
)
from .routing_policies import (
    router as _routing_policies_router,
)
from .runs import (
    ComparisonRequest,
    create_comparison,
    download_artifact,
    event_runs,
    list_runs,
    realtime_token,
)
from .runs import (
    router as _runs_router,
)

router = APIRouter(prefix="/v1/studio", tags=["studio"])
for _sub_router in (
    _extract_router,
    _datasets_router,
    _model_profiles_router,
    _dynamic_schemas_router,
    _routing_policies_router,
    _benchmark_router,
    _deploy_router,
    _runs_router,
):
    router.include_router(_sub_router)
del _sub_router

__all__ = [
    "router",
    "_shared",
    "_DOMAIN_404",
    "DEFAULT_TOPICS",
    "EXTRACT_EVENT",
    "BENCHMARK_EVENT",
    "DEPLOY_EVENT",
    "SEED_EVENT",
    "MODELS_CONFIG_PATH",
    "TriggerResponse",
    "_record_event_owners",
    "_annotate_fits",
    "_node_available_bytes",
    "ExtractRequest",
    "trigger_extract",
    "RenderDocumentRequest",
    "render_document",
    "list_datasets",
    "validate_dataset_version",
    "list_model_profiles",
    "PipelineProfileRequest",
    "create_pipeline_profile",
    "OcrProfileRequest",
    "create_ocr_profile",
    "delete_pipeline_profile_route",
    "list_dynamic_schemas_route",
    "get_dynamic_schema_route",
    "create_dynamic_schema_route",
    "delete_dynamic_schema_route",
    "RoutingPolicyCreateRequest",
    "list_routing_policies_route",
    "get_routing_policy_route",
    "create_routing_policy_route",
    "delete_routing_policy_route",
    "BenchmarkRequest",
    "trigger_benchmark",
    "DeployRequest",
    "trigger_deploy",
    "SeedHfRequest",
    "trigger_seed_hf",
    "hf_repo_ggufs",
    "hf_search",
    "hf_inspect",
    "hf_collection",
    "SeedOllamaRequest",
    "trigger_seed_ollama",
    "realtime_token",
    "event_runs",
    "list_runs",
    "ComparisonRequest",
    "create_comparison",
    "download_artifact",
]
