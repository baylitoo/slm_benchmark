# Decisions

Design decisions are made in GitHub issues, which keep the alternatives and why
they lost. These pages do not re-narrate them. This index points at the ones
that shape how the system works today; a decision that needed more than its
issue to record has its own page.

```{toctree}
:maxdepth: 1

agent-sandboxing
```

| Decision | Where | Issue |
|---|---|---|
| Model-generated code runs in Judge0, not in-process | {doc}`agent-sandboxing`, [#263](https://github.com/baylitoo/slm_benchmark/issues/263) | closed |
| Distributed serving: one control plane, N serving nodes | [#527](https://github.com/baylitoo/slm_benchmark/issues/527), {doc}`../architecture/serving` | open |
| Remote lifecycle operations need a desired state that can express them | [#590](https://github.com/baylitoo/slm_benchmark/issues/590) | open |
| Session affinity is computed by rendezvous hashing, not remembered | [#337](https://github.com/baylitoo/slm_benchmark/issues/337), {doc}`../architecture/serving` | closed |
| Parallel extraction splits by top-level list, bounded by observed slots | [#444](https://github.com/baylitoo/slm_benchmark/issues/444), {doc}`../architecture/extraction` | open |
| Dynamic schemas have no bare scalar list, so every leaf carries evidence | [#424](https://github.com/baylitoo/slm_benchmark/issues/424) | open |
| Saved schemas are edited in place, without versions | [#460](https://github.com/baylitoo/slm_benchmark/issues/460) | open |

New decisions get an issue first — see {doc}`../development/contributing`.
