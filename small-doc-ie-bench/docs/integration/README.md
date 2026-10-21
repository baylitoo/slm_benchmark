# Integration contracts

Documents handed to a *partner team* who integrates against this platform, as
opposed to `docs/` proper, which documents the platform for the people building
it. They are written to be read by someone outside this repository, so they
state the status of what they describe and avoid assuming any familiarity with
the codebase.

LaTeX rather than Markdown because these go out as PDFs: they are sent to
another company's team, reviewed, annotated and signed off. The `.tex` source is
what lives here; the PDF is a build artifact and is not committed.

## Building

Plain `pdflatex`, two passes (the second resolves cross-references). No packages
beyond a standard TeX Live or MiKTeX install, so they also compile unchanged in
Overleaf.

```sh
cd docs/integration
pdflatex -interaction=nonstopmode realtime-stt-contract.fr.tex
pdflatex -interaction=nonstopmode realtime-stt-contract.fr.tex
```

The French version uses `babel[french]`, which applies French typography
automatically — the thin space before `:` and `;`, and a localised date.

## Contents

| File | Audience | Status |
|---|---|---|
| `realtime-stt-contract.en.tex` | ERP integrator, English | **Proposal — not implemented** |
| `realtime-stt-contract.fr.tex` | ERP integrator, French | **Proposal — not implemented** |

The two are the same document in two languages; keep them in step when either
changes.

### Real-time speech-to-text

Covers the browser-microphone dictation flow between the Cynov ERP and this
platform: ephemeral ticket minting, the WebSocket handshake, audio format, and
the message protocol.

It describes a design that **does not exist yet** and says so on its first page.
The purpose is to agree the contract with the other team before either side
builds against it, so nothing in it should be treated as an API reference until
the implementation lands and the status box is removed.

Design discussion and the open engine/latency questions are in the
*Real-time streaming transcription* issue; this document is deliberately only
the part the integrator needs, with our internal open points summarised rather
than argued.
