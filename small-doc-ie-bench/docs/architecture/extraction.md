# Extraction pipeline

Turning a document into structured, validated, evidence-linked data. The path
is linear; most of the complexity sits in two places — how a schema becomes
something a model can be constrained to, and what happens when the model does
not cooperate.

## The path

```{mermaid}
flowchart LR
    A[Document] --> B{Profile kind}
    B -->|vision| C[Rasterise pages]
    B -->|text| D[OCR backend]
    C --> E[Build prompt]
    D --> E
    E --> F[Model call]
    F --> G[Parse + normalise]
    G --> H[Rehydrate]
    H --> I[Ground to blocks]
    I --> J[Validate]
    J --> K[Response]
```

**Ingestion** branches on the resolved profile. A vision profile rasterises PDF
pages to images at a configured DPI, rejecting documents over a page limit. A
text profile runs an OCR backend, cached by document hash plus backend and
configuration. Supplied text skips both and becomes a single block.

**Prompt building** is chosen by the profile's prompt profile — a NuExtract
variant, a vision prompt, or the generic strict-extraction prompt, which renders
OCR blocks into the user message.

**The model call** goes through the gateway (see below), which owns
concurrency, retries and the circuit breaker.

**Parsing** cleans the content, repairs bare keys and mojibake, and parses JSON.
NuExtract families get a normalisation pass first.

**Rehydration** re-wraps the model's flat output into the rich internal shape —
each field becoming a value plus evidence ids plus confidence.

**Grounding** matches extracted values back to the OCR blocks they came from, so
a field can be shown with its source rather than asserted.

**Validation** runs the Pydantic model. It returns errors; it does not raise.
An invalid result is still returned, with the problems attached — the caller
decides what to do about a document that did not parse cleanly.

## Schemas, and the two shapes

A schema exists in two forms, and conflating them causes confusion.

The **rich** schema is the internal one: every field is an object carrying the
value, the evidence ids that support it, and a confidence. This is what gets
validated and audited.

The **flat** schema is what the model sees. `flatten_schema_json` resolves
references, unwraps those triples back to plain values, and strips the keywords
that break grammar compilation — patterns, bounds, formats. Without that step
llama.cpp cannot compile the schema into a grammar at all.

So the model is asked for something simple, and the platform reconstitutes the
rich structure around it afterwards.

Schemas come from two places. **Static** schemas are Pydantic models in the
registry. **Dynamic** schemas are a field spec — strings, dates, numbers, money,
objects, lists — built into a Pydantic model at runtime. When a dynamic
extraction is requested without a spec, the platform asks a model to propose
one from the document's text first.

## Profiles and families

A **model profile** is the resolved answer to "how do I call this model":
endpoint, model id, sampling parameters, timeouts, concurrency and queue limits,
prompt profile, vision settings, and the response-format style.

A **family contract** is what makes a profile possible for a model the code has
never seen. It carries the response-format style, the prompt profile, how the
template is delivered, extra server arguments, and whether the model does
vision, tools or embeddings. Families are why deploying a new model of a known
kind needs no new code.

## The negotiation ladder

Not every server supports every way of constraining output. Rather than
requiring the operator to know which, the client **walks a ladder**:

```
openai_json_schema  →  json_object  →  none
llamacpp_schema     →  json_object  →  none
```

A rung is abandoned when the response is empty or unparseable, or when the
server rejects the request with a grammar compilation error. The effective
style ends up on the response, so what actually worked is visible rather than
guessed.

```{important}
Only **generic** styles downgrade. Purpose-built styles — the NuExtract ones,
the vLLM guided modes — are singleton ladders and are never silently rewritten
into something else. Downgrading them would quietly produce a different task.
```

Two extra recovery rungs sit in the same attempt sequence:

- A retry **without the assistant prefill** after a grammar error, when prefill
  was used.
- One **reasoning-disabled retry** when the model hit its length limit with no
  usable content — the budget went into hidden reasoning rather than an answer.

## Parallel extraction

Opt-in per request. When enabled, the schema is split so each top-level list
field becomes its own group, with the remaining scalar and object fields forming
one shared group. Each group is a separate constrained call against a narrowed
schema, and the results are merged.

It is skipped when streaming is active, or when the schema is too small to split
usefully.

Concurrency is bounded by the deployment's **observed slot count** — the real
`/slots` number the reconciler measured — capped by the profile's own limit.
A profile with no slot information runs one at a time.

```{note}
`deployment_slot_count` is only populated on `store:` profiles. A bare
deployment or config profile has none, so parallel extraction degrades to
sequential rather than guessing a width.
```

A group failing fails the whole extraction. There is no partial result.

## Failure modes

The gateway classifies errors rather than treating them alike:

| Condition | Treatment |
|---|---|
| `429` | rate limited, honours `Retry-After` |
| `408`, `409`, `425`, `5xx` | transient, retried with backoff |
| other `4xx` | not retried |
| read timeout | **not retried** |
| queue full or wait exceeded | rejected immediately |
| repeated failures | circuit opens, half-open probe after a delay |

The read-timeout rule is deliberate and worth internalising: retrying a request
that timed out while reading would queue a duplicate prompt behind one that is
still running, making the overload worse. Connection-level timeouts *are*
retried, because nothing is running on the other end.

For responses, the distinctions are similarly specific:

- **Truncated JSON at the length limit** raises immediately. No weaker style or
  retry fixes it — the answer did not fit, and the fix is more tokens or
  splitting the schema.
- **Unparseable content** walks the ladder down, and only raises on the last
  rung.
- **Validation failure** does not raise at all. The result comes back marked
  invalid with the errors attached.

Gateway state — concurrency, circuit, queue — is **per process**, keyed by
endpoint and model. With several API replicas, each keeps its own, which means
the effective limits are multiplied by the replica count. The same caveat
applies to tenant quotas.
