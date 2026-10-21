"""Preconfigured agent templates the Studio catalog offers.

A template is a starting point: ``defaults`` prefill the create form / request
body and remain fully overridable. Templates are code, not data — adding one
is a PR, so the catalog stays reviewed and reproducible.
"""

from __future__ import annotations

from typing import Any

from docie_bench.agents.pii import PII_TYPES

AGENT_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "proxy-security",
        "kind": "proxy_security",
        "display_name": "Security Proxy Agent",
        "description": (
            "Privacy firewall in front of any served SLM: detects PII "
            "(emails, phones, IBANs, cards, national ids, IPs) and anonymizes "
            "it with stable placeholders before the model sees the prompt. "
            "Modes: placeholder (mask), block (refuse), detect (annotate only). "
            "Also the seam for IP/confidentiality screening."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {
                "entities": list(PII_TYPES),
                "mode": "placeholder",
                "restore_pii": False,
                # Encoder-family analyzer (e.g. a `docie encoder` GLiNER
                # deployment). When set it replaces the regex analyzer;
                # guard_labels/guard_threshold tune it, guard_fallback: "regex"
                # opts into degraded analysis when the guard is down
                # (fail-closed otherwise).
                "guard_model": None,
                "guard_labels": None,
                "guard_threshold": None,
                "guard_fallback": None,
                # Moderation presets for GLiNER2 guardrail checkpoints (e.g.
                # fastino/GLiNER2-Guardrails-PII-Multi): prompt_safety,
                # prompt_toxicity, jailbreak_detection. In block mode a
                # non-benign verdict refuses the request (unsafe_blocked).
                "guard_tasks": None,
            },
        },
    },
    {
        "id": "ocr-agent",
        "kind": "ocr",
        "display_name": "OCR Agent",
        "description": (
            "Instantiates an OCR solution behind an OpenAI endpoint: send a "
            "document as an inline image_url data URI, get its text back. Set "
            "an extractor profile (e.g. a NuExtract deployment) to turn it "
            "into an OCR→SLM structured-extraction pipeline."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {"backend": "tesseract", "language": None, "extractor": None},
        },
    },
    {
        "id": "custom",
        "kind": "custom",
        "display_name": "Custom Agent",
        "description": (
            "Bring your own: a system prompt over any served model. The "
            "starting point for building new agents on the platform."
        ),
        "defaults": {"system_prompt": "", "options": {}},
    },
    {
        "id": "docs-search-agent",
        "kind": "custom",
        "display_name": "Document Search Agent",
        "description": (
            "Agentic RAG demo: wires a custom agent to the Document Search "
            "MCP server (enable it under MCP servers first — see the "
            "catalog) so even a small model searches a directory of "
            "documents before answering, instead of guessing from a "
            "stuffed prompt. Every tool call is visible in Try it (#262) "
            "and the Observability usage ledger (#261)."
        ),
        "defaults": {
            "system_prompt": (
                "You are a document search assistant. You do not know the "
                "contents of any document from memory. Before answering, "
                "call list_files to see what is available, then use "
                "search_text or read_document to find the actual answer. "
                "Cite which document and page your answer came from. If "
                "nothing relevant is found, say so plainly instead of "
                "guessing."
            ),
            "options": {"mcp_servers": ["docs-search"], "mcp_tools": None},
        },
    },
    # --- Low-latency (encoder-family) templates -------------------------
    #
    # These three exist to make the encoder models usable without reading the
    # architecture docs first. All of them run a DeBERTa-class encoder
    # (0.2-0.4B) on CPU in tens of milliseconds, which is a different
    # operating point from a generative SLM: no tokens are generated, so
    # latency does not scale with how much you ask for.
    #
    # The shared limitation is worth stating once: an encoder extractor COPIES
    # SPANS. It cannot compute a total, normalise "le 12 mars 2019" into a
    # date, or infer a value that is not written down. Fields like those need
    # a generative model, which is why the first template keeps the extractor
    # slot overridable rather than pretending one model covers everything.
    {
        "id": "low-latency-extraction",
        "kind": "ocr",
        "display_name": "Low-latency Extraction",
        "description": (
            "OCR plus an encoder extractor (GLiFormer / GLiNER2), end to end "
            "on CPU in tens of milliseconds. Point `extractor` at a deployed "
            "encoder and `schema` at the document type; the schema's field "
            "names become the labels the model fills from spans it finds. "
            "Because every value is a span, the evidence offsets are exact "
            "rather than fuzzy-matched after the fact, which is what the "
            "Review tab needs. Use it for fields written verbatim on the "
            "page -- numbers, dates as printed, names, line items. Computed "
            "or normalised fields (a total that must be summed, a date that "
            "must be reformatted) need a generative extractor instead."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {
                "backend": "tesseract",
                "language": None,
                # A deployed encoder, e.g. a `docie encoder` GLiFormer
                # deployment. Empty so the create form makes you choose one
                # that actually exists on this node.
                "extractor": "",
                "schema": "invoice",
            },
        },
    },
    {
        "id": "document-triage",
        "kind": "custom",
        "display_name": "Document Triage",
        "description": (
            "Zero-shot multi-label classification on a deployed GLiClass "
            "encoder: one CPU pass answers 'what kind of document is this', "
            "scoring every candidate label together rather than once per "
            "label as an NLI classifier would. Labels come from the request "
            "(`labels`) or from the deployment's defaults, so new categories "
            "need no retraining. Built to sit in front of extraction, where "
            "the caller should not have to know the document type in "
            "advance. It classifies only -- it locates nothing and fills no "
            "schema."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {},
        },
    },
    {
        "id": "low-latency-pii-guard",
        "kind": "proxy_security",
        "display_name": "Low-latency PII Guard",
        "description": (
            "The security proxy with its encoder analyzer switched on: a "
            "GLiNER-family guard replaces the regex detector, so PII is "
            "found by meaning rather than by pattern -- names and addresses, "
            "which no regex catches, and fewer false positives on strings "
            "that merely look like an id. Point `guard_model` at a deployed "
            "encoder. With a GLiNER2 guardrails checkpoint, `guard_tasks` "
            "additionally screens for unsafe prompts and jailbreaks in the "
            "same pass. Fails closed by default: if the guard is "
            "unreachable the request is refused rather than quietly "
            "falling back to regex -- set `guard_fallback` to opt into "
            "degraded analysis."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {
                "entities": list(PII_TYPES),
                "mode": "placeholder",
                "restore_pii": False,
                # Empty rather than None: this template exists to be pointed
                # at an encoder, so the form should prompt for one.
                "guard_model": "",
                "guard_labels": None,
                "guard_threshold": None,
                "guard_fallback": None,
                "guard_tasks": None,
            },
        },
    },
    {
        "id": "workflow-agent",
        "kind": "workflow",
        "display_name": "Workflow Agent",
        "description": (
            "Prompt chaining (#265): a fixed, ORDERED sequence of steps, "
            "each its own backing model and system prompt. One request "
            "runs the whole sequence server-side — each step after the "
            "first receives only the previous step's answer, not the "
            "original request. Built to prove small (350M-class) models "
            "handle narrow sub-tasks reliably in a pipeline even where one "
            "model doing the whole thing in one shot would struggle."
        ),
        "defaults": {
            "system_prompt": None,
            "options": {
                "steps": [
                    {
                        "model_profile": "",
                        "system_prompt": (
                            "Extract the key facts from the input as a short "
                            "bullet list. Output only the bullets."
                        ),
                    },
                    {
                        "model_profile": "",
                        "system_prompt": (
                            "Using only the bullet list you were given, write "
                            "a one-paragraph summary."
                        ),
                    },
                ]
            },
        },
    },
]


def template_by_id(template_id: str) -> dict[str, Any] | None:
    for template in AGENT_TEMPLATES:
        if template["id"] == template_id:
            return template
    return None
