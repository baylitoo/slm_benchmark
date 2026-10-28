# DocIE

A document information-extraction platform. It deploys and supervises small
language models on its own hardware, exposes them behind an OpenAI-compatible
API, and runs document extraction as durable jobs with a Studio to operate the
whole thing.

This documentation is written for the people building and running it.

```{toctree}
:caption: Architecture
:maxdepth: 1

architecture/overview
architecture/inngest
architecture/serving
architecture/extraction
architecture/auth
```

```{toctree}
:caption: Operations
:maxdepth: 1

operations/getting-started
operations/studio
operations/security
operations/fleet-runbook
```

```{toctree}
:caption: Models
:maxdepth: 1

models/onboarding
models/reasoning
```

```{toctree}
:caption: Development
:maxdepth: 1

development/contributing
development/testing
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/glossary
reference/api
reference/serving-cli
reference/embedding-models
decisions/index
```

## Where to start

To run it, start with {doc}`operations/getting-started`. To change it, read
{doc}`architecture/overview` first — it is the map, and the other pages assume
it.

After that it depends what you are doing:

- **Changing how jobs run** — {doc}`architecture/inngest`. Everything
  long-running goes through there, and the role split is the first thing that
  confuses people.
- **Changing how models are served** — {doc}`architecture/serving`. The most
  intricate part of the system, and the one with the most invariants.
- **Changing what extraction produces** — {doc}`architecture/extraction`.
- **Anything touching credentials** — {doc}`architecture/auth`. Three
  credential kinds that are deliberately not interchangeable.
- **Running a fleet of serving nodes** — {doc}`operations/fleet-runbook`.
  Start there before adding a second machine; most of what can be
  misconfigured fails as a wrong answer rather than an error.
- **Integrating an application** — {doc}`reference/api`. Use a Studio-issued
  key, not an operator key.
- **Adding a model the platform has not seen** — {doc}`models/onboarding`.
- **Your first change** — {doc}`development/contributing`, then
  {doc}`development/testing`.

When a word seems to mean two things, it probably does:
{doc}`reference/glossary`.

## Scope of these pages

Design decisions live in GitHub issues rather than being re-narrated here. The
issue carries the alternatives and why they lost, which is the part that is
expensive to reconstruct and impossible to read out of a diff. Where a page
describes a decision, it links to the issue that made it; {doc}`decisions/index`
lists the ones that shape the system today.
