"""Evaluators: judges that score what DocIE produced, after the fact.

A judge reads one ``EvalSubject`` (a rendered agent run, for now) and returns
``Verdict``s. Two kinds share that interface:

* ``DecisionJudge`` asks a decision model on llama-server's ``/v1/systemone``.
  Its probabilities are read off a classifier head: ``provenance="measured"``.
* ``LlmJudge`` prompts a chat model for a JSON verdict. Its score is text the
  model wrote: ``provenance="self_reported"``.

The two are never merged into one confidence number; callers that compare them
must look at ``provenance``.
"""

from docie_bench.evaluators.judges import (
    DecisionJudge,
    EvalSubject,
    Judge,
    JudgeQuestion,
    LlmJudge,
    Verdict,
    render_transcript,
)

__all__ = [
    "DecisionJudge",
    "EvalSubject",
    "Judge",
    "JudgeQuestion",
    "LlmJudge",
    "Verdict",
    "render_transcript",
]
