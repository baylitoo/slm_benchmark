# Security

The platform processes adversarial documents — invoices, identity documents,
OCR text, uploads — for several tenants. A document is never trusted as an
instruction. What is being protected is tenant data, model credentials,
availability, audit records, and the host filesystem.

How callers are authenticated is in {doc}`../architecture/auth`. This page is
the operator's side: the boundaries, what defends each, what to set in
production, and what is left over.

## Trust boundaries

| Boundary | |
|---|---|
| Client → API | `X-API-Key` when `AUTH_REQUIRED=true` |
| API → OCR and model servers | document content stays untrusted on the far side |
| API → audit database | stores tenant context, hashes, and redacted results |
| Benchmark API → host filesystem | off by default, because it accepts server-side paths |
| Agents → MCP tool servers | only servers an operator registered, by name; never a caller-supplied URL or command. The process runs with the server's own privileges, so registering one is an operator trust decision |
| Agents → code execution | the `code_interpreter` tool runs each snippet in Judge0, a separate sandboxed service on the internal network with no published port and a required token — never in-process |

## Defences

| Threat | Defence |
|---|---|
| Oversized uploads or text | bounded streaming reads; configurable text and OCR-block limits |
| Spoofed extension or `Content-Type` | extension allowlist plus magic-byte validation |
| Prompt injection in documents | explicit untrusted-data boundaries in every extraction, proposer, vision and judge prompt |
| One tenant exhausting another | per-tenant request and concurrency limits |
| PII in logs and audits | document content logging off by default; recursive field redaction for audits and responses |
| Arbitrary server-side paths | benchmark API off by default |
| Uploads left on disk | temporary files removed in `finally`; raw document storage off by default |

## Production settings

```text
AUTH_REQUIRED=true
API_KEYS={"long-random-key-a":"tenant-a"}
RATE_LIMIT_REQUESTS=60
TENANT_READ_RATE_LIMIT_REQUESTS=600
RATE_LIMIT_WINDOW_SECONDS=60
TENANT_MAX_CONCURRENT_REQUESTS=4
MAX_UPLOAD_MB=25
MAX_REQUEST_BODY_MB=26
MAX_TEXT_CHARS=1000000
REDACTED_AUDIT_FIELDS=vendor_tax_id,customer_tax_id,iban,document_number,mrz_line_1,mrz_line_2
LOG_DOCUMENT_CONTENT=false
ENABLE_BENCHMARK_API=false
FORWARDED_ALLOW_IPS=<proxy network>
```

`API_KEYS` is for operators. Applications get Studio-issued keys.

Beyond the file: terminate TLS in front of the API, keep keys in a secret
manager, and put the model and OCR services on a private network. The Inngest
dashboard has no authentication, so it belongs on that network too.

## Residual risk

- **Prompt boundaries reduce injection; they do not prevent it.** Schema
  validation and evidence grounding remain the controls that matter.
- **MIME validation is not a parser sandbox.** For hostile public uploads, run
  PDF and image parsing in resource-limited isolated processes.
- **The early body-size limit relies on `Content-Length`.** Cap chunked bodies
  at the ingress proxy.
- **Quotas are per process** and reset on restart; several API replicas
  multiply them.
- **Redaction matches field names.** A sensitive field under an unexpected name
  needs configuration.
- **Nothing here claims regulatory compliance,** and nothing protects data once
  it reaches an external model endpoint.
