# Graph Report - .  (2026-07-02)

## Corpus Check
- Large corpus: 203 files · ~822,959 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 2191 nodes · 5563 edges · 100 communities (89 shown, 11 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 1205 edges (avg confidence: 0.7)
- Token cost: 233,247 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Declarative Routing Policy|Declarative Routing Policy]]
- [[_COMMUNITY_Experiment Orchestration API|Experiment Orchestration API]]
- [[_COMMUNITY_Dataset Loading & Registry|Dataset Loading & Registry]]
- [[_COMMUNITY_Interactive TUI (cli2)|Interactive TUI (cli2)]]
- [[_COMMUNITY_Tenant Auth & Quota|Tenant Auth & Quota]]
- [[_COMMUNITY_ModelArtifact Manifests|Model/Artifact Manifests]]
- [[_COMMUNITY_Report & Reproducibility|Report & Reproducibility]]
- [[_COMMUNITY_Docker Compose Stack|Docker Compose Stack]]
- [[_COMMUNITY_NuExtract3 Normalization|NuExtract3 Normalization]]
- [[_COMMUNITY_Ollama Profile Generation|Ollama Profile Generation]]
- [[_COMMUNITY_Run Comparison & Budgets|Run Comparison & Budgets]]
- [[_COMMUNITY_Serving Defaults (PlannerSupervisor)|Serving Defaults (Planner/Supervisor)]]
- [[_COMMUNITY_GGUF Model Store|GGUF Model Store]]
- [[_COMMUNITY_Capability Probe|Capability Probe]]
- [[_COMMUNITY_Extraction & Review API|Extraction & Review API]]
- [[_COMMUNITY_Control Plane Core|Control Plane Core]]
- [[_COMMUNITY_OCR Cache|OCR Cache]]
- [[_COMMUNITY_Control Plane Operations|Control Plane Operations]]
- [[_COMMUNITY_OCR Backend Interface|OCR Backend Interface]]
- [[_COMMUNITY_OpenAI-Compatible Client|OpenAI-Compatible Client]]
- [[_COMMUNITY_ResourceRuntime Planner|Resource/Runtime Planner]]
- [[_COMMUNITY_Judge Calibration Gate|Judge Calibration Gate]]
- [[_COMMUNITY_Inngest Durable Functions|Inngest Durable Functions]]
- [[_COMMUNITY_Solution Adapters|Solution Adapters]]
- [[_COMMUNITY_Review DB Models|Review DB Models]]
- [[_COMMUNITY_Remote Runtime Adapter|Remote Runtime Adapter]]
- [[_COMMUNITY_Embeddable Serving CLI|Embeddable Serving CLI]]
- [[_COMMUNITY_Benchmark UI Components|Benchmark UI Components]]
- [[_COMMUNITY_LlamaCppOllama Runtimes|LlamaCpp/Ollama Runtimes]]
- [[_COMMUNITY_Benchmark Runner & Metrics|Benchmark Runner & Metrics]]
- [[_COMMUNITY_Gateway Model Routing|Gateway Model Routing]]
- [[_COMMUNITY_Field Metrics & Alignment|Field Metrics & Alignment]]
- [[_COMMUNITY_Postgres Model Catalog|Postgres Model Catalog]]
- [[_COMMUNITY_Frontend API Client|Frontend API Client]]
- [[_COMMUNITY_Review Workflow Logic|Review Workflow Logic]]
- [[_COMMUNITY_Persistent Supervisor|Persistent Supervisor]]
- [[_COMMUNITY_Model Gateway (Circuit Breaker)|Model Gateway (Circuit Breaker)]]
- [[_COMMUNITY_Supervisor Tests|Supervisor Tests]]
- [[_COMMUNITY_Runtime Tests|Runtime Tests]]
- [[_COMMUNITY_Studio Run Store|Studio Run Store]]
- [[_COMMUNITY_Artifact Blob Store|Artifact Blob Store]]
- [[_COMMUNITY_TypeScript Config|TypeScript Config]]
- [[_COMMUNITY_Gateway Execution & Capabilities|Gateway Execution & Capabilities]]
- [[_COMMUNITY_Extraction Observability & Review Enqueue|Extraction Observability & Review Enqueue]]
- [[_COMMUNITY_Evidence Grounding & Prompts|Evidence Grounding & Prompts]]
- [[_COMMUNITY_Studio API (Jobs)|Studio API (Jobs)]]
- [[_COMMUNITY_Model Store Tests|Model Store Tests]]
- [[_COMMUNITY_Unified OpenAI Gateway|Unified OpenAI Gateway]]
- [[_COMMUNITY_Result Panels UI|Result Panels UI]]
- [[_COMMUNITY_Frontend Build Config|Frontend Build Config]]
- [[_COMMUNITY_App Shell & Navigation|App Shell & Navigation]]
- [[_COMMUNITY_Response Negotiation Tests|Response Negotiation Tests]]
- [[_COMMUNITY_Solution Adapter Tests|Solution Adapter Tests]]
- [[_COMMUNITY_Settings & Extraction Entry|Settings & Extraction Entry]]
- [[_COMMUNITY_ValidityConstrained Gate|Validity/Constrained Gate]]
- [[_COMMUNITY_Model Gateway Tests|Model Gateway Tests]]
- [[_COMMUNITY_Voxel51 Invoice Scans|Voxel51 Invoice Scans]]
- [[_COMMUNITY_Layout & Toast Providers|Layout & Toast Providers]]
- [[_COMMUNITY_Deploy UI & Hooks|Deploy UI & Hooks]]
- [[_COMMUNITY_Serving Read API|Serving Read API]]
- [[_COMMUNITY_Negotiation Benchmark Test|Negotiation Benchmark Test]]
- [[_COMMUNITY_Realtime Publish|Realtime Publish]]
- [[_COMMUNITY_Schema Validators|Schema Validators]]
- [[_COMMUNITY_Sample Dataset Documents|Sample Dataset Documents]]
- [[_COMMUNITY_Playground UI|Playground UI]]
- [[_COMMUNITY_Annotation Export API|Annotation Export API]]
- [[_COMMUNITY_Validity Gate & CPU Sampler|Validity Gate & CPU Sampler]]
- [[_COMMUNITY_UI Utility Components|UI Utility Components]]
- [[_COMMUNITY_Runs Query Endpoints|Runs Query Endpoints]]
- [[_COMMUNITY_Voxel51 Manifest Builder|Voxel51 Manifest Builder]]
- [[_COMMUNITY_Field Grounding|Field Grounding]]
- [[_COMMUNITY_Control Plane CLI|Control Plane CLI]]
- [[_COMMUNITY_Studio Store Defaults|Studio Store Defaults]]
- [[_COMMUNITY_Extraction Strategy Concepts|Extraction Strategy Concepts]]
- [[_COMMUNITY_Frontend Runtime Deps|Frontend Runtime Deps]]
- [[_COMMUNITY_Studio Index Models|Studio Index Models]]
- [[_COMMUNITY_CORS Config|CORS Config]]
- [[_COMMUNITY_Test State Resets|Test State Resets]]
- [[_COMMUNITY_Judge Evaluation Tests|Judge Evaluation Tests]]
- [[_COMMUNITY_Benchmark Config Concepts|Benchmark Config Concepts]]
- [[_COMMUNITY_LLM Judge|LLM Judge]]
- [[_COMMUNITY_Value Normalization|Value Normalization]]
- [[_COMMUNITY_OCR Artifact Model|OCR Artifact Model]]
- [[_COMMUNITY_DB Engine Startup|DB Engine Startup]]
- [[_COMMUNITY_GGUF Downloader|GGUF Downloader]]
- [[_COMMUNITY_Inngest Client|Inngest Client]]
- [[_COMMUNITY_Inngest Package Init|Inngest Package Init]]
- [[_COMMUNITY_Test Fixtures (Quota)|Test Fixtures (Quota)]]
- [[_COMMUNITY_Serving Factory Concept|Serving Factory Concept]]
- [[_COMMUNITY_Next.js Config|Next.js Config]]
- [[_COMMUNITY_PostCSS Config|PostCSS Config]]
- [[_COMMUNITY_Tailwind Config|Tailwind Config]]
- [[_COMMUNITY_Orchestration Package|Orchestration Package]]
- [[_COMMUNITY_Store Package Init|Store Package Init]]
- [[_COMMUNITY_Repo Root|Repo Root]]

## God Nodes (most connected - your core abstractions)
1. `ModelProfile` - 68 edges
2. `run_benchmark()` - 54 edges
3. `RuntimeLaunchSpec` - 48 edges
4. `PersistentSupervisor` - 47 edges
5. `OCRBlock` - 46 edges
6. `ExtractionService` - 43 edges
7. `ControlPlane` - 42 edges
8. `ModelStore` - 40 edges
9. `OrchestratorService` - 38 edges
10. `ExtractionResponse` - 37 edges

## Surprising Connections (you probably didn't know these)
- `test_score_prediction_nested_value()` --calls--> `score_prediction()`  [INFERRED]
  tests/test_metrics.py → src/docie_bench/benchmark/metrics.py
- `test_score_prediction_threads_evidence_applicable_without_touching_field_score()` --calls--> `score_prediction()`  [INFERRED]
  tests/test_metrics.py → src/docie_bench/benchmark/metrics.py
- `test_score_evidence_reports_coverage_and_ungrounded_fields()` --calls--> `score_evidence()`  [INFERRED]
  tests/test_metrics.py → src/docie_bench/benchmark/metrics.py
- `test_score_evidence_vision_path_is_not_applicable()` --calls--> `score_evidence()`  [INFERRED]
  tests/test_metrics.py → src/docie_bench/benchmark/metrics.py
- `test_both_mode_reports_judge_ground_truth_calibration_delta()` --calls--> `summarize()`  [INFERRED]
  tests/test_judge_evaluation.py → src/docie_bench/benchmark/runner.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Deterministic Regression Gate Flow** — _github_workflows_regression_gate_workflow, configs_regression_budgets_budgets, readme_judge_calibration_gate [INFERRED 0.80]
- **CPU Cascade Routing Pattern** — configs_routing_policy_example_cascade, configs_models_ollama_qwen25_1b, configs_models_ollama_nuextract_3b, readme_extraction_router [EXTRACTED 1.00]
- **DocIE Studio async stack (web/api/inngest/worker)** — docker_compose_web, docker_compose_api, docker_compose_inngest, docker_compose_worker, docs_docie_studio [EXTRACTED 0.95]
- **ControlPlane injected collaborators** — docs_serving_factory_control_plane, docs_serving_factory_registry, docs_serving_factory_runtime_catalog, docs_serving_factory_supervisor, docs_serving_factory_planner [EXTRACTED 0.95]
- **Observability metrics pipeline (api -> prometheus -> grafana)** — docker_compose_api, docker_compose_prometheus, docker_compose_grafana, infra_prometheus_prometheus, infra_grafana_provisioning_datasources_datasource [EXTRACTED 0.95]
- **Batch1 invoices share the same seller/client/items/summary template** — data_voxel51_invoices_data_batch1_0001_invoice, data_voxel51_invoices_data_batch1_0002_invoice, data_voxel51_invoices_data_batch1_0003_invoice, data_voxel51_invoices_data_batch1_0004_invoice, data_voxel51_invoices_data_batch1_0005_invoice [INFERRED 0.95]
- **Batch1 scans participate in the Voxel51 invoice dataset** — data_voxel51_invoices_data_batch1_0001_invoice, data_voxel51_invoices_data_batch1_0002_invoice, data_voxel51_invoices_data_batch1_0003_invoice, data_voxel51_invoices_data_batch1_0004_invoice, data_voxel51_invoices_data_batch1_0005_invoice [INFERRED 0.95]
- **Batch1 invoices sharing seller/client VAT layout** — data_voxel51_invoices_data_batch1_0006, data_voxel51_invoices_data_batch1_0007, data_voxel51_invoices_data_batch1_0008, data_voxel51_invoices_data_batch1_0009, data_voxel51_invoices_data_batch1_0010 [INFERRED 0.95]

## Communities (100 total, 11 thin omitted)

### Community 0 - "Declarative Routing Policy"
Cohesion: 0.05
Nodes (94): BaseModel, build_extraction_router(), load_routing_policy(), Path, Bridge declarative routing policies to the benchmark's model profiles.  The ro, Load and validate a declarative routing policy from YAML or JSON., Return the distinct profiles a policy references, in first-seen stage order., Build a router whose stages map 1:1 to model profiles by name. (+86 more)

### Community 1 - "Experiment Orchestration API"
Cohesion: 0.06
Nodes (68): BenchmarkRun, BenchmarkTask, Event, Executor, cancel_experiment(), claim_task(), complete_task(), configure_orchestrator() (+60 more)

### Community 2 - "Dataset Loading & Registry"
Cohesion: 0.05
Nodes (84): exists, help, min, Option, readable, DatasetItem, load_dataset(), Path (+76 more)

### Community 3 - "Interactive TUI (cli2)"
Cohesion: 0.05
Nodes (52): App, DataTable(), Header, Pressed, Screen, _ask_dataset(), _ask_options(), _ask_profiles() (+44 more)

### Community 4 - "Tenant Auth & Quota"
Cohesion: 0.07
Nodes (57): detect_mime_type(), get_quota_manager(), parse_api_keys(), UploadFile, Process-wide tenant quota manager, built once from settings.      Single sourc, FastAPI dependency: authenticate the caller, then bound per-tenant quota., read_validated_upload(), tenant_guard() (+49 more)

### Community 5 - "Model/Artifact Manifests"
Cohesion: 0.09
Nodes (33): BinaryIO, ArtifactKind, ArtifactManifest, ArtifactVerificationError, ModelConflictError, ModelManifest, ModelNotFoundError, ModelRegistry (+25 more)

### Community 6 - "Report & Reproducibility"
Cohesion: 0.10
Nodes (54): _cpu_chart_svg(), Any, Path, write_report(), append_jsonl(), atomic_write_json(), atomic_write_text(), canonical_json() (+46 more)

### Community 7 - "Docker Compose Stack"
Cohesion: 0.06
Nodes (51): api service (FastAPI), artifact-store shared volume, bench service (docie-bench tool), grafana service, inngest service (server + Connect gateway), llm-llamacpp service (local GGUF server), postgres service, prometheus service (+43 more)

### Community 8 - "NuExtract3 Normalization"
Cohesion: 0.08
Nodes (37): Image, LogRecord, _derive_invoice_subtotal(), _norm_amount(), _norm_date(), _normalize_nested_nuextract(), _normalize_nuextract_raw(), Post-process NuExtract3 output: enforce document_type, strip IBAN spaces,     n (+29 more)

### Community 9 - "Ollama Profile Generation"
Cohesion: 0.08
Nodes (41): append_profile(), build_profile_config(), default_profile_name(), detect_capabilities(), ModelCapabilities, _ollama_api_host(), Any, Path (+33 more)

### Community 10 - "Run Comparison & Budgets"
Cohesion: 0.15
Nodes (39): _compare_observations(), compare_runs(), ComparisonResult, _confidence_interval(), _evaluate_budgets(), _group_observations(), _group_payload(), list_baselines() (+31 more)

### Community 11 - "Serving Defaults (Planner/Supervisor)"
Cohesion: 0.11
Nodes (14): Protocol, Result, _DefaultPlanner, _DefaultRuntimes, _DefaultSupervisor, Planner, Any, Build the local control plane from the serving implementation modules. (+6 more)

### Community 12 - "GGUF Model Store"
Cohesion: 0.11
Nodes (28): _assert_within(), _blob_path(), default_ollama_home(), _entry_from_json(), FamilyContract, get_family(), _link_or_copy(), ModelStore (+20 more)

### Community 13 - "Capability Probe"
Cohesion: 0.13
Nodes (34): _advertised_metadata(), cached_probe_for_endpoint(), CapabilityProbe, get_cached_probe(), probe_endpoint(), profile_probe_fingerprint(), Any, Runtime capability probe for the response-format negotiation path.  A profile (+26 more)

### Community 14 - "Extraction & Review API"
Cohesion: 0.12
Nodes (32): File, Form, approve_review_task(), claim_review_task(), correct_review_task(), create_review(), default_profile(), enforce_request_content_length() (+24 more)

### Community 15 - "Control Plane Core"
Cohesion: 0.13
Nodes (18): Enum, ControlPlane, Coordinate registry, runtime, deployment, and planning operations., RuntimeKind, FakePlanner, FakeRegistry, FakeRuntimes, FakeSupervisor (+10 more)

### Community 16 - "OCR Cache"
Cohesion: 0.13
Nodes (19): OCRArtifact, Portable, versioned output shared by OCR backends, cache, and benchmarks., OCRCache, Any, Path, Content-addressed OCR artifact cache with atomic writes and corruption checks., hash_file(), OCRProcessor (+11 more)

### Community 17 - "Control Plane Operations"
Cohesion: 0.11
Nodes (11): _DefaultRegistry, _optional(), Path, T, Control-plane facade for model-serving operations.  The facade deliberately de, Recursively convert common backend values to deterministic JSON-safe data., _replicas(), _required() (+3 more)

### Community 18 - "OCR Backend Interface"
Cohesion: 0.08
Nodes (16): ABC, OCRBackend, Any, Path, Return a cache-relevant backend/runtime version., Return cache-relevant backend configuration., stable_block_id(), text_to_blocks() (+8 more)

### Community 19 - "OpenAI-Compatible Client"
Cohesion: 0.08
Nodes (23): ModelCapabilities, _clean_content(), _fix_bare_keys(), OpenAICompatibleClient, Any, Response-format styles to try, strongest confirmed rung first.          The ru, Issue one minimal completion with a single style; report if honored., Normalise raw LLM output to a single JSON object string.      Handles:     - (+15 more)

### Community 20 - "Resource/Runtime Planner"
Cohesion: 0.17
Nodes (21): check_runtime_compatibility(), PlanningRequest, Any, ModelManifest, Conservative runtime compatibility and resource recommendation planner., recommend_runtime(), ResourcePlanner, RuntimeName (+13 more)

### Community 21 - "Judge Calibration Gate"
Cohesion: 0.15
Nodes (27): calibration_gate(), compute_judge_agreement(), _dimension_agreement(), evaluate_calibration(), _evaluate_dimension(), load_calibration(), Any, Path (+19 more)

### Community 22 - "Inngest Durable Functions"
Cohesion: 0.11
Nodes (28): benchmark_idempotency_key(), deploy_model_job(), extract_document(), _gc_studio_runs(), gc_studio_runs_job(), _gc_studio_runs_sync(), Any, Path (+20 more)

### Community 23 - "Solution Adapters"
Cohesion: 0.13
Nodes (22): build_solution(), _chat_completion(), _decode_data_uri(), _extract_document(), _inject_ocr_text(), _ocr_to_text(), OcrSolution, PipelineSolution (+14 more)

### Community 24 - "Review DB Models"
Cohesion: 0.18
Nodes (25): DeclarativeBase, ReviewConflictError, ReviewError, ReviewNotFoundError, ReviewValidationError, _validate_correction_paths(), FieldCorrection, Base (+17 more)

### Community 25 - "Remote Runtime Adapter"
Cohesion: 0.17
Nodes (9): RuntimeError, Path, RemoteRuntime, RuntimeCapabilities, RuntimeConfigurationError, RuntimeLaunchError, RuntimeLaunchSpec, RuntimeProcess (+1 more)

### Community 26 - "Embeddable Serving CLI"
Cohesion: 0.11
Nodes (12): create_app(), Create an embeddable CLI, optionally backed by an injected control plane., FakePlane, test_backend_is_constructed_lazily(), test_commands_delegate_to_the_control_plane(), test_human_list_output_is_a_readable_table(), test_json_errors_are_machine_readable_and_exit_nonzero(), test_json_output_is_compact_deterministic_and_preserves_backend_order() (+4 more)

### Community 27 - "Benchmark UI Components"
Cohesion: 0.10
Nodes (16): BADGE_TONES, BTN_SIZES, BTN_VARIANTS, ButtonSize, ButtonVariant, ComingSoon(), DOT_TONES, EmptyState() (+8 more)

### Community 28 - "LlamaCpp/Ollama Runtimes"
Cohesion: 0.11
Nodes (12): HealthGet, PopenFactory, RunCommand, command_display(), _default_health_get(), default_runtime_adapters(), LlamaCppRuntime, OllamaRuntime (+4 more)

### Community 29 - "Benchmark Runner & Metrics"
Cohesion: 0.12
Nodes (21): EvaluationMode, BenchmarkResult, BenchmarkTask, _constrained_style_stats(), _hallucination_semantics(), _percentile(), Any, Aggregate the per-row effective response-format style for one profile.      Re (+13 more)

### Community 30 - "Gateway Model Routing"
Cohesion: 0.11
Nodes (18): Exception, GatewayRoutingError, A requested model could not be routed to exactly one upstream., Map a requested model to exactly one profile (name first, then upstream id)., resolve_profile(), _client(), _profile(), Request (+10 more)

### Community 31 - "Field Metrics & Alignment"
Cohesion: 0.21
Nodes (23): _align_rows(), compare_values(), _decimal_or_none(), _evidence_fields(), _evidence_rows(), _flatten_row(), get_path(), _greedy_alignment() (+15 more)

### Community 32 - "Postgres Model Catalog"
Cohesion: 0.16
Nodes (17): available_backends(), CatalogUnavailableError, ModelCatalog, ModelStoreEntry, Any, datetime, Postgres-backed catalog of the local GGUF model store.  Blobs live on disk (the, Backends that can serve a model of ``family`` faithfully.      llama-server can (+9 more)

### Community 33 - "Frontend API Client"
Cohesion: 0.13
Nodes (21): BenchmarkRequest, DeploymentRecord, deployModel(), DeployRequest, detailOf(), getBenchmarks(), getModels(), getRealtimeToken() (+13 more)

### Community 34 - "Review Workflow Logic"
Cohesion: 0.36
Nodes (21): _as_utc(), _assert_claim_owner(), _assert_version(), claim_review(), correct_review(), decide_review(), _event(), _expire_task_claim() (+13 more)

### Community 35 - "Persistent Supervisor"
Cohesion: 0.24
Nodes (10): LifecycleState, DeploymentRecord, PersistentSupervisor, Any, Re-run reconcile() until the deployment is READY or the timeout elapses., Small single-node desired-state reconciler with durable JSON state., _record_from_dict(), _record_to_dict() (+2 more)

### Community 36 - "Model Gateway (Circuit Breaker)"
Cohesion: 0.21
Nodes (16): CircuitOpenError, classify_response_error(), ErrorClassification, _GatewayState, InvalidModelResponseError, ModelCapabilityError, ModelGatewayError, ModelQueueFullError (+8 more)

### Community 37 - "Supervisor Tests"
Cohesion: 0.25
Nodes (15): HealthResult, _deployment(), FakeAdapter, Any, Path, test_await_ready_polls_until_healthy(), test_await_ready_returns_last_state_on_timeout(), test_corrupt_state_is_rejected() (+7 more)

### Community 38 - "Runtime Tests"
Cohesion: 0.17
Nodes (15): VLLMRuntime, FakeProcess, Any, MonkeyPatch, Path, _spec(), test_is_running_uses_psutil_for_pids_not_in_process_table(), test_llamacpp_requires_gguf_and_builds_cpu_flags() (+7 more)

### Community 39 - "Studio Run Store"
Cohesion: 0.24
Nodes (11): StudioRun, StudioRunArtifact, _artifact_to_dict(), Any, Durable index for Studio benchmark runs (Postgres + blob store)., Reserve a run row *before* doing any work.          Returns ``("claimed", reco, Owning tenant of an event, or ``None`` if no ownership is recorded.          C, Resolve the dedup key to emit for a fresh benchmark trigger.          A genuin (+3 more)

### Community 40 - "Artifact Blob Store"
Cohesion: 0.15
Nodes (10): ArtifactBlobStore, Path, Session, sessionmaker, Resolve a store-relative key to an absolute path *inside* the root.          G, Delete a blob and prune now-empty digest directories. Idempotent., Yield the store-relative key of every committed blob under the root., Last-modified time of a blob (UTC), or ``None`` if it is gone. (+2 more)

### Community 41 - "TypeScript Config"
Cohesion: 0.10
Nodes (19): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+11 more)

### Community 42 - "Gateway Execution & Capabilities"
Cohesion: 0.20
Nodes (6): Never, ModelCapabilities, ModelGateway, Any, Exception, T

### Community 43 - "Extraction Observability & Review Enqueue"
Cohesion: 0.15
Nodes (17): finalize_response(), _confidence_values(), enqueue_review(), _evidence_counts(), Any, score_review_candidate(), _set_path(), ReviewReason (+9 more)

### Community 44 - "Evidence Grounding & Prompts"
Cohesion: 0.21
Nodes (16): ground_evidence(), Link typed extraction fields to the best matching OCR block., build_nuextract3_prompts(), build_nuextract_prompts(), build_schema_proposer_prompt(), build_user_prompt(), Return (system_prompt, user_prompt) in NuExtract **v1** format.      NuExtract, Return (system_prompt, user_prompt) for NuExtract3.      NuExtract3 receives t (+8 more)

### Community 45 - "Studio API (Jobs)"
Cohesion: 0.22
Nodes (18): BenchmarkRequest, DeployRequest, download_artifact(), ExtractRequest, Response, TenantDependency, FastAPI glue for DocIE Studio: trigger jobs + consume results.  Mounted on the, Stream a run artifact (``report.html`` / ``predictions.jsonl`` / ``metrics.json` (+10 more)

### Community 46 - "Model Store Tests"
Cohesion: 0.23
Nodes (17): ModelStoreError, _fake_ollama_home(), Path, Containment must NOT reject legitimate refs that contain '/' and ':'., Build a minimal Ollama models dir: one manifest + content-addressed blobs., A crafted reference must not read manifests outside the manifests root., A crafted store name must not write blobs outside the store root., test_library_reference_resolves_to_registry_ollama_ai() (+9 more)

### Community 47 - "Unified OpenAI Gateway"
Cohesion: 0.16
Nodes (17): AsyncBaseTransport, FastAPI, JSONResponse, create_gateway_app(), _dispatch_solution(), _forward_stream(), _openai_error(), AsyncClient (+9 more)

### Community 48 - "Result Panels UI"
Cohesion: 0.16
Nodes (14): PollingResult(), AnyToken, RealtimeResult(), stateTone(), Mode, ResultPanel(), Badge(), BadgeTone (+6 more)

### Community 49 - "Frontend Build Config"
Cohesion: 0.11
Nodes (17): devDependencies, autoprefixer, postcss, tailwindcss, @types/node, @types/react, @types/react-dom, typescript (+9 more)

### Community 50 - "App Shell & Navigation"
Cohesion: 0.15
Nodes (10): AppShell(), HEALTH_META, NAV, SectionId, Benchmark(), Observability(), ThemeToggle(), Card() (+2 more)

### Community 51 - "Response Negotiation Tests"
Cohesion: 0.32
Nodes (16): _chat(), _client(), _completion(), _grammar_error(), _profile(), Any, Response, Response-format negotiation: the downgrade ladder in ``chat_json``.  These tes (+8 more)

### Community 52 - "Solution Adapter Tests"
Cohesion: 0.26
Nodes (14): _Block, _client(), fake_ocr(), _FakeBackend, _image_request(), _ocr_profile(), _png_data_uri(), MonkeyPatch (+6 more)

### Community 53 - "Settings & Extraction Entry"
Cohesion: 0.18
Nodes (7): BaseSettings, hash_file(), Any, Path, build_vision_user_prompt(), get_settings(), Settings

### Community 54 - "Validity/Constrained Gate"
Cohesion: 0.26
Nodes (14): evaluate_constrained_gate(), evaluate_validity_gate(), Return the summary rows whose ``valid_rate`` is below ``threshold``.      A no, Return summary rows whose ``constrained_rate`` fell below ``threshold``., _constrained_row(), Validity gate: a below-threshold valid_rate must fail loudly, not score zero., _row(), test_constrained_gate_disabled_when_threshold_non_positive() (+6 more)

### Community 55 - "Model Gateway Tests"
Cohesion: 0.35
Nodes (14): _chat(), _client(), _completion(), _profile(), Any, Request, Response, test_capability_discovery_rejects_missing_model_and_unsupported_format() (+6 more)

### Community 56 - "Voxel51 Invoice Scans"
Cohesion: 0.34
Nodes (14): Invoice document, Seller/Client VAT invoice layout, Seller/Client + Items + Summary invoice layout (10% VAT), Voxel51 invoice dataset, Invoice 51109338 (Andrews, Kirby and Valdez → Becker Ltd), Invoice 12847181 (Fitzpatrick and Sons → Duncan PLC), Invoice 19471831 (Palmer Ltd → Rios, Oneill and Rowe), Invoice 16273983 (Reyes, Holloway and Lee → Castillo LLC) (+6 more)

### Community 57 - "Layout & Toast Providers"
Cohesion: 0.16
Nodes (10): metadata, Providers(), ICONS, Toast, ToastContext, ToastContextValue, ToastInput, ToastProvider() (+2 more)

### Community 58 - "Deploy UI & Hooks"
Cohesion: 0.22
Nodes (10): Deploy(), formatBytes(), getDeployments(), getFamilies(), getStore(), StoreEntry, AsyncState, useAsync() (+2 more)

### Community 59 - "Serving Read API"
Cohesion: 0.25
Nodes (13): _control_plane(), deployment_status(), list_benchmarks(), list_deployments(), list_families(), list_models(), list_runtimes(), list_store() (+5 more)

### Community 60 - "Negotiation Benchmark Test"
Cohesion: 0.30
Nodes (13): _completion(), _downgrade_handler(), _inputs(), _install_stub_transport(), Any, MonkeyPatch, Path, Request (+5 more)

### Community 61 - "Realtime Publish"
Cohesion: 0.17
Nodes (12): @inngest/realtime, metrics(), Response, Run a benchmark and persist its artifacts to the durable store.      Returns a, Run a full benchmark over a dataset and persist addressable artifacts.      Ev, _run_benchmark_job(), publish(), Any (+4 more)

### Community 62 - "Schema Validators"
Cohesion: 0.32
Nodes (11): _collect_evidence_ids(), _money_amount(), _number_value(), Any, BaseModel, Decimal, validate_extraction(), _validate_invoice_arithmetic() (+3 more)

### Community 63 - "Sample Dataset Documents"
Cohesion: 0.17
Nodes (12): Versioned Dataset Registry, sample Dataset v1.0.0, ID Card 001 (French CNI), ID Card 002 (German Personalausweis), Invoice 001 (ACME SAS, FR), Invoice 002 (TechPro Solutions, FR), Invoice 003 (Brenner & Associates, DE), Invoice 004 (Northfield Consulting, GB) (+4 more)

### Community 64 - "Playground UI"
Cohesion: 0.18
Nodes (10): DeployForm(), SeedForm(), InputMode, Playground(), useToast(), Button, TextArea, ApiError (+2 more)

### Community 65 - "Annotation Export API"
Cohesion: 0.20
Nodes (11): export_review_annotations(), export_annotations(), Path, AnnotationExportRequest, AnnotationExportView, ClaimRequest, CorrectionRequest, ReleaseRequest (+3 more)

### Community 66 - "Validity Gate & CPU Sampler"
Cohesion: 0.17
Nodes (6): Raised when a profile's valid_rate falls below the configured threshold., ValidityGateError, CpuSampler, Samples system-wide CPU% every `interval` seconds in a background thread., test_raising_the_gate_is_a_runtime_error(), test_validity_gate_error_message_lists_failures()

### Community 67 - "UI Utility Components"
Cohesion: 0.36
Nodes (7): RuntimeChip(), JsonView(), safeStringify(), LiveIndicator(), timeAgo(), StatusDot(), cn()

### Community 68 - "Runs Query Endpoints"
Cohesion: 0.22
Nodes (10): ge, le, min_length, Query, event_runs(), list_runs(), Any, Run status for an event.      Benchmark runs have a durable index row (metrics (+2 more)

### Community 69 - "Voxel51 Manifest Builder"
Cohesion: 0.31
Nodes (9): build_ground_truth(), main(), normalize_amount(), normalize_quantity(), Convert the Voxel51 high-quality-invoice-images dataset into a docie-bench manif, Return a canonical 'NNNN.NN' string from a US- or EU-formatted amount., Normalize a quantity; drop a trailing .00 so '3,00' -> '3'., Convert MM/DD/YYYY to ISO YYYY-MM-DD, or None if empty/unparseable. (+1 more)

### Community 70 - "Field Grounding"
Cohesion: 0.49
Nodes (9): _best_match(), _candidate_variants(), _copy_and_ground(), _copy_and_ground_row(), _field_candidate(), _match_score(), _normalize(), Any (+1 more)

### Community 71 - "Control Plane CLI"
Cohesion: 0.42
Nodes (8): _cell(), _Context, _execute(), _json(), Ollama-like operations CLI for the serving control plane.  Run with ``python -, _render(), _render_rows(), _state()

### Community 72 - "Studio Store Defaults"
Cohesion: 0.28
Nodes (8): default_blob_store(), default_run_store(), _isoformat(), datetime, Blob store + run index service for durable Studio benchmark results.  ``Artifa, A blob committed to the store, addressed by its store-relative key., Build a RunStore from process defaults (shared blob dir + app DB)., StoredBlob

### Community 73 - "Extraction Strategy Concepts"
Cohesion: 0.25
Nodes (8): nuextract3 Model Profile, ollama_nuextract_3b Model Profile, ollama_qwen25_1b Model Profile, CPU Cascade Routing Policy, Multi-stage Extraction Router, Human Review Workflow, Schema-constrained Extraction, Vision Extraction Path

### Community 74 - "Frontend Runtime Deps"
Cohesion: 0.25
Nodes (8): dependencies, clsx, lucide-react, next, next-themes, react, react-dom, tailwind-merge

### Community 75 - "Studio Index Models"
Cohesion: 0.29
Nodes (6): datetime, SQLAlchemy models for the durable Studio run index.  One ``StudioRun`` row per, Lightweight event id -> triggering principal binding.      Recorded for every, StudioEventOwner, utcnow(), Bind an event id to its triggering principal (idempotent).          Recorded f

### Community 76 - "CORS Config"
Cohesion: 0.43
Nodes (6): parse_cors_origins(), Parse STUDIO_CORS_ORIGINS into an allow-origins list.      Empty/unset falls b, test_comma_separated_override_is_parsed_and_trimmed(), test_default_origins_are_localhost_not_wildcard(), test_empty_or_blank_falls_back_to_default(), test_explicit_wildcard_is_preserved()

### Community 77 - "Test State Resets"
Cohesion: 0.33
Nodes (7): Clear the shared probe cache. Intended for tests and reconfiguration., reset_probe_cache(), Clear shared scheduler/circuit and capability-probe state.      Intended for t, reset_gateway_state(), _reset(), _reset_gateway(), _reset_gateway()

### Community 78 - "Judge Evaluation Tests"
Cohesion: 0.38
Nodes (6): Path, test_both_mode_reports_judge_ground_truth_calibration_delta(), test_cli_accepts_unlabeled_document_in_judge_mode(), test_load_judge_profile_uses_separate_config_selector(), test_runner_evaluates_unlabeled_document_and_excludes_judge_profile(), test_summarize_includes_judge_metrics_without_ground_truth()

### Community 79 - "Benchmark Config Concepts"
Cohesion: 0.33
Nodes (6): Regression Gate CI Workflow, Benchmark Configuration, Regression Budgets, Hallucination Rate by Ingestion Path, Judge Calibration Gate, OCR/Layout Backends

### Community 80 - "LLM Judge"
Cohesion: 0.67
Nodes (5): build_judge_prompt(), judge_extraction(), Any, _score(), test_judge_extraction_sends_source_and_adds_model_metadata()

### Community 81 - "Value Normalization"
Cohesion: 0.40
Nodes (3): normalize_decimal(), Any, Decimal

### Community 82 - "OCR Artifact Model"
Cohesion: 0.47
Nodes (4): OCRPageImage, OCRQualitySignals, quality_signals(), test_common_artifact_supports_page_images_and_quality_signals()

### Community 83 - "DB Engine Startup"
Cohesion: 0.50
Nodes (5): startup(), get_session_factory(), init_engine(), Session, sessionmaker

## Knowledge Gaps
- **97 isolated node(s):** `metadata`, `SectionId`, `NAV`, `HEALTH_META`, `InputMode` (+92 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_benchmark()` connect `Report & Reproducibility` to `Declarative Routing Policy`, `Dataset Loading & Registry`, `Validity Gate & CPU Sampler`, `Interactive TUI (cli2)`, `Ollama Profile Generation`, `Capability Probe`, `Extraction & Review API`, `Judge Evaluation Tests`, `Realtime Publish`, `Settings & Extraction Entry`, `Validity/Constrained Gate`, `Negotiation Benchmark Test`, `Benchmark Runner & Metrics`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Why does `ModelProfile` connect `Model Gateway (Circuit Breaker)` to `Declarative Routing Policy`, `Dataset Loading & Registry`, `Report & Reproducibility`, `NuExtract3 Normalization`, `Ollama Profile Generation`, `Capability Probe`, `Extraction & Review API`, `OpenAI-Compatible Client`, `Inngest Durable Functions`, `Solution Adapters`, `Benchmark Runner & Metrics`, `Gateway Model Routing`, `Gateway Execution & Capabilities`, `Unified OpenAI Gateway`, `Response Negotiation Tests`, `Solution Adapter Tests`, `Model Gateway Tests`, `Validity Gate & CPU Sampler`, `Judge Evaluation Tests`, `LLM Judge`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `ControlPlane` connect `Control Plane Core` to `Persistent Supervisor`, `Interactive TUI (cli2)`, `Control Plane CLI`, `Serving Defaults (Planner/Supervisor)`, `GGUF Model Store`, `Model Store Tests`, `Control Plane Operations`, `Resource/Runtime Planner`, `Remote Runtime Adapter`, `Embeddable Serving CLI`, `Serving Read API`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 30 inferred relationships involving `ModelProfile` (e.g. with `EvaluationMode` and `ResumeDriftError`) actually correct?**
  _`ModelProfile` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 47 inferred relationships involving `run_benchmark()` (e.g. with `run_benchmark_endpoint()` and `DatasetItem`) actually correct?**
  _`run_benchmark()` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `RuntimeLaunchSpec` (e.g. with `ControlPlane` and `_DefaultPlanner`) actually correct?**
  _`RuntimeLaunchSpec` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 33 inferred relationships involving `PersistentSupervisor` (e.g. with `ControlPlane` and `.from_defaults()`) actually correct?**
  _`PersistentSupervisor` has 33 INFERRED edges - model-reasoned connections that need verification._