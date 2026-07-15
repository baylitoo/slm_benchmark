"""Encoder model family — specialized token classifiers behind the OpenAI spec.

Encoders (GLiNER-style zero-shot NER, PII detectors, confidentiality
classifiers) are not chat models, but they are served behind the SAME
OpenAI-compatible surface as everything else in this framework: text goes in
as a user message, the detected entities come back as the assistant's JSON
content. That keeps the whole serving contract uniform — the security proxy
agent, the gateway, and any external platform consume an encoder deployment
exactly like an SLM deployment, and only the payload convention differs.

``docie encoder`` launches the reference server (GLiNER backend, CPU-friendly,
``pip install .[encoders]``); anything else that honours the same convention
(a remote endpoint, another shim) plugs in identically via a models.yaml
profile or a deployment record.
"""

from docie_bench.encoders.server import create_encoder_app

__all__ = ["create_encoder_app"]
