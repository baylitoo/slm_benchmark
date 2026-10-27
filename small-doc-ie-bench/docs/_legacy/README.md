# Superseded documentation

These pages were written incrementally alongside the work they describe. They
contain accurate facts, but they were never written as documentation: they
assume the context of the change that produced them, overlap each other, and
have no consistent shape.

They are kept here, excluded from the published site, while their content is
absorbed into the structured documentation one section at a time. A page is
deleted from this directory once what it knows is covered elsewhere.

Do not link to these from the site, and do not edit them — fix the replacement
page instead.

## Status

| Page | Replaced by |
|---|---|
| `serving-control-plane.md`, `serving-factory.md` | `architecture/serving.md` |
| `model-onboarding.md`, `lfm25-reasoning.md` | a models section, deliberately deferred |
| `adr-agent-sandboxing.md` | the decision index |
| `server-integration.md` | a reference page for the `/v1` surface |

`server-integration.md` is the one to be careful with: it is a contract another
team integrates against. Its replacement has to exist before it is removed.
