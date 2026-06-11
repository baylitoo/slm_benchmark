# Threat Model

## Scope

The API processes adversarial invoices, identity documents, OCR text, metadata, and file
uploads for multiple tenants. Documents are never trusted as instructions. The primary assets
are tenant data, model credentials, service availability, audit records, and host filesystem
access.

## Trust Boundaries

- Internet clients to FastAPI: authenticate with `X-API-Key` when `AUTH_REQUIRED=true`.
- FastAPI to OCR and model servers: document content remains untrusted across this boundary.
- API process to audit database: tenant context, hashes, and configured/redacted results are stored.
- Benchmark API to host filesystem: disabled by default because it accepts server-side paths.

## Defenses

| Threat | Defense |
| --- | --- |
| Oversized uploads or OCR/text payloads | Bounded streaming upload reads and configurable text/block limits |
| File extension or `Content-Type` spoofing | Extension allowlist plus magic-byte/content validation |
| Prompt injection in documents or metadata | Explicit untrusted-data instructions and boundaries in extraction, schema proposer, NuExtract, vision, and judge prompts |
| Cross-tenant quota exhaustion | API-key-derived tenant context, per-tenant rate limits, and concurrent request limits |
| Unauthenticated multi-tenant use | Optional mandatory API-key authentication with constant-time key comparison |
| PII leakage in logs and audits | Full prompts/OCR/model output logging disabled by default; configurable recursive audit and response redaction |
| Arbitrary server-side benchmark paths | Benchmark API disabled by default |
| Temporary upload persistence | Temporary file removed in `finally`; raw document storage remains disabled by default |

## Configuration

Production deployments should set at least:

```dotenv
AUTH_REQUIRED=true
API_KEYS={"long-random-key-a":"tenant-a","long-random-key-b":"tenant-b"}
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
TENANT_MAX_CONCURRENT_REQUESTS=4
MAX_UPLOAD_MB=25
MAX_REQUEST_BODY_MB=26
MAX_TEXT_CHARS=1000000
REDACTED_AUDIT_FIELDS=vendor_tax_id,customer_tax_id,iban,document_number,mrz_line_1,mrz_line_2
REDACTED_RESPONSE_FIELDS=
LOG_DOCUMENT_CONTENT=false
ENABLE_BENCHMARK_API=false
```

Terminate TLS before the API, keep API keys in a secret manager, isolate the model/OCR
services on a private network, and use an external distributed rate limiter when running more
than one API process.

## Residual Risk

- Prompt instructions reduce injection risk but cannot guarantee model compliance. Schema
  validation and evidence grounding remain mandatory controls.
- MIME validation identifies supported formats but is not antivirus or a full parser sandbox.
  Run OCR/PDF/image parsing in resource-limited isolated processes for hostile public uploads.
- The early whole-request limit depends on `Content-Length`. Reject or cap chunked request bodies
  at the ingress proxy.
- Quotas are in-memory and apply per process. They do not coordinate across replicas and reset
  on restart.
- API keys are static bearer credentials. There is no rotation endpoint, OAuth, or fine-grained
  authorization.
- Redaction is field-name based. Unexpected sensitive fields require configuration updates.
- Existing databases need a migration to add the nullable/indexed `tenant_id` audit column.
- The API does not claim regulatory compliance or protect data after it reaches an external
  model endpoint.
