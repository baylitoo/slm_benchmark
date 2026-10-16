# Testing

123 test modules cover the backend; the frontend is tested with Vitest. Neither
suite calls a live model, and that is a constraint rather than an accident.

## Running them

```sh
# backend — PYTHONPATH matters in a worktree
PYTHONPATH=src python -m pytest tests/test_serving_runtime.py -q

# frontend
cd frontend && npx vitest run lib/auth.test.ts
cd frontend && npx tsc --noEmit
```

Run the tests for what you touched, not the whole suite, during ordinary work.
CI runs everything; locally, a full run mostly buys waiting.

## What a good test looks like here

**No live model calls.** A test that needs a model uses a fake backend. The
value of the suite is that it runs in seconds on any machine, and one network
call forfeits that.

**Realistic names.** Use the models this project actually serves —
`lfm2.5-350m`, `nuextract3` — not `qwen-7b` placeholders. Test fixtures are
read as documentation of what the system handles.

**A regression test must fail without the fix.** Write it, revert the fix,
watch it fail, restore the fix. A test added alongside a change that would have
passed before it is not a regression test; it is decoration. This has caught
real mistakes — including a migration ordering bug where every existing test
passed because test databases are always created fresh.

That last case is worth generalising: **if a bug depends on pre-existing
state, the test has to construct that state explicitly.** A suite that only
ever builds new databases cannot see a migration that breaks old ones.

**One or two tests per fix.** Enough to pin the behaviour, not a matrix.

## The fake-backend trap

Several suites inject a fake model backend. A fake that accepts any argument
will happily accept a call the real library would reject — which once let a
500-level bug through a green suite.

When you fake a library boundary, make the fake enforce the real contract: if
the library raises on a combination of arguments, the fake raises too.

## Guard tests

A few tests exist to make a silent failure loud rather than to check a feature:

- **Route inventory.** `tests/test_studio_api_route_inventory.py` pins the
  exact `(method, path)` surface of the Studio API. A router that is built but
  never mounted compiles fine and serves 404 for an entire domain; this is the
  net for that. It also asserts a route count, so adding a route means updating
  two places on purpose.
- **Family and profile synthesis.** Pins what a deployment's defaults resolve
  to, so a change to a family contract cannot quietly re-shape every request.

If one of these fails, read it before changing it. They fail in exactly the
situations where the change looked harmless.

## Frontend

Component tests use Testing Library; `lib/` tests are plain unit tests over the
API clients and credential handling.

`tsc --noEmit` is part of the gate, and `npm run build` catches what `tsc` does
not — a production build exercises the route tree and the server/client
boundary.

Check the exit code of the tool, not of the last pipe stage. `npx tsc ... | tail`
reports `tail`'s status, which is always zero.

## What CI runs

The regression gate runs the backend suite, lint and types. Documentation has
its own workflow which builds this site with warnings treated as errors, so a
broken cross-reference or a page missing from a toctree fails the build rather
than shipping quietly.
