# Glossary

Several words in this codebase are overloaded, and the overloading is the main
source of confusion for a new reader. This page fixes the meanings.

```{glossary}
deployment
  A model runtime that has been asked to exist: a name, a model, launch
  parameters, and a desired state. Confusingly it means three subtly different
  things depending on where you are standing — a *record* in `deployments.json`
  (desired), a *process* supervised by the serving node (actual), and a *row*
  in the placement table (observed). When precision matters, say which.

placement
  The observed row for a deployment in the Postgres catalog: where it is, what
  state it is in, whether it is healthy, how fast it has measured, and how many
  slots it has. A placement is what the resolver routes to; a deployment record
  is what the reconciler tries to make true.

desired state
  What should be running. Currently a node-local file (`deployments.json`),
  single-writer.

observed state
  What is actually running, as last seen by the reconciler and published to the
  catalog. Never inferred — a deployment is reported healthy only on a real
  health probe.

reconciler
  The loop inside the serving role that compares desired against observed,
  repairs the difference within a restart budget, and publishes what it saw.

profile
  `ModelProfile` — the resolved answer to "how do I call this model": endpoint,
  model id, timeouts, token limits, and the family contract that shapes the
  request. Produced by a resolver, never written by hand at call time.

family
  A contract shared by models of the same kind. Carries things like the
  response-format style, the prompt profile, and whether the model does vision,
  tools, embeddings or reranking. It is why the extraction path can send a
  correctly-shaped request to a model it has never seen.

model profile (the selector)
  Confusingly also the *string a caller sends* to choose a model. It accepts a
  config profile name, an upstream model id, a live deployment name, or
  `store:<name>`. See {doc}`../architecture/serving`.

slot
  One concurrent request a runtime can serve, configured as `n_parallel`.
  Slots within one process share the weights — and the memory bandwidth — so
  more slots buy concurrency, not throughput.

store model
  A model in the platform's own catalog, addressed as `store:<name>` and
  resolved through Postgres placements rather than through a node's local file.
  This is the path that already works across machines.

tenant
  The principal a request is attributed to. Comes from the API key's configured
  tenant, or from the signed-in person's company login. It is the dimension
  usage, quotas and ownership are keyed on — not a hard data boundary.

operator
  A caller authenticated with an env-configured `API_KEYS` key. Unscoped: may
  manage keys for any tenant.

studio user
  A caller authenticated with a Studio session token, proved by a company
  sign-in. May use the Studio and manage their own keys, nothing else.

serving role
  A container running with `DOCIE_WORKER_ROLE=serving`: the one process that
  holds the model subprocess handles and runs the reconciler. Deliberately
  single-replica.

worker role
  A container running `DOCIE_WORKER_ROLE=worker`: the scaled Inngest workers
  that run extraction and benchmark jobs. Holds no model processes.

control plane
  The part of the system that decides what should run and makes it so —
  deployment lifecycle, supervision, placement. As opposed to the data plane,
  which is the request path that uses what is running.

grounding
  Linking an extracted value back to where it came from in the source document,
  so a field can be shown with evidence rather than asserted.

OCR blocks
  Text fragments with positions, produced by an OCR backend or supplied by the
  caller. The unit the prompt is built from, and what grounding matches against.

real-time factor (RTF)
  Processing seconds divided by audio or document seconds. Below 1 means faster
  than real time. The measure that decides whether streaming work is viable on
  a given host.
```

## Things that sound like each other

**Deployment vs placement.** Desired vs observed. If you are reading
`deployments.json` you are looking at intent; if you are reading the catalog
you are looking at reality.

**Profile vs family.** A family is the contract for a *kind* of model; a
profile is the resolved configuration for *one* call to *one* model.

**Operator vs studio user.** Both are authenticated people. Only the operator
can act outside their own tenant.

**Slot vs replica.** A slot is concurrency inside one process, sharing one copy
of the weights. A replica is another process with its own copy. They cost very
different amounts of memory and buy different things.
