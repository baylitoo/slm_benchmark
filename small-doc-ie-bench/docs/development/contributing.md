# Contributing

## Branches and pull requests

Every change goes through a branch and a pull request. Nothing is pushed to the
integration branch directly.

Branch from the remote tip, not from a local copy of it:

```sh
git fetch origin
git checkout -b fix-the-thing origin/dev-agents-milestone
```

A local integration branch does not fast-forward on `fetch`, so branching from
it is how you end up building on a tree that is weeks behind.

**Do not stack pull requests.** If your work depends on an unmerged branch,
wait for it to land and then open against the updated base. Stacked PRs have
repeatedly produced merge orders that looked fine and silently dropped changes.

After any merge, verify the **tree**, not the badge — `git grep` for the thing
that was supposed to land. A PR can report merged while its content is not where
you expect it.

### Worktrees are mandatory

Many agents and sessions share this repository. Never edit files or switch
branches in the shared main checkout:

```sh
git worktree add .claude/worktrees/my-change -b my-change origin/dev-agents-milestone
```

Two consequences worth knowing up front:

- `pytest` in a worktree will import `docie_bench` from the **main** checkout
  unless you set the path. Always run `PYTHONPATH=src python -m pytest`.
- A fresh worktree has no `frontend/node_modules`. Link it from the main
  checkout before running `vitest` or `tsc` rather than reinstalling.

The shared stash stack is also shared with other sessions. Prefer a temporary
commit to setting work aside; if you must stash, tag it and recover it by SHA.

## Writing the change

**Match the surrounding code.** Comment density, naming and idiom vary by
module; the local style wins over any global preference.

**Comment the why, not the what.** The codebase is deliberately heavy on
explanation where a decision is non-obvious — why a check exists, what breaks
without it, what was tried and rejected — and light everywhere else. A comment
restating the line below it is noise; a comment recording why the line is not
the obvious one is the most valuable thing in the file.

**Prefer the generic fix.** Per-document or per-model tuning is rejected on
sight. If a bug reproduces on one invoice, the fix belongs at the level that
makes it impossible for a class of documents, not that one.

**Audit what already exists before adding a dependency.** This codebase
repeatedly found that the capability it needed was already present and unused in
something it already ships.

## Before you commit

```sh
git diff --cached
```

Read it. A passing local test run is not proof of what is staged; this has
caused real CI failures.

Run, as a minimum, the tests for the files you touched, plus lint and types on
exactly those files:

```sh
PYTHONPATH=src python -m pytest tests/test_the_thing.py -q
python -m ruff check <changed files>
PYTHONPATH=src python -m mypy <changed files>
```

Scope lint to your own files. The tree carries a small number of pre-existing
findings, and a diff that fixes unrelated ones buries the change under noise.

## Commit and PR text

Describe the artifact, not the session that produced it. No narrative about
what was tried, no references to the conversation, no "as requested".

Pull request bodies follow the shape of a good bug report:

> **Bug** — what is wrong, with the error if it is short.
> **Reproduce** — the minimal path to seeing it.
> **Root cause** — why, with the anchor.
> **Fix** — what changed and why that is the right level.
> **Testing** — what was run, and honestly what was not.

State what you did **not** verify. A PR that says "tests pass, deployment not
exercised" is more useful than one that implies more than it checked.

## Tests

See {doc}`testing`. The short version: one or two targeted tests per fix, no
live model calls, and a regression test must be shown to fail without the fix.

## Design decisions get an issue

Anything that changes the shape of the system — a new boundary, a new
credential kind, a change to how models are addressed — gets a GitHub issue
before it gets a pull request. The issue carries the alternatives and why they
lost, which is the part that is expensive to reconstruct later and impossible to
read out of the diff.

The architecture pages here deliberately point at those issues rather than
re-narrating them.
