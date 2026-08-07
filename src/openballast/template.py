"""Prompt templates.

The research-measured conditions (logprob scoring) are completion-style:

    PROMPT_TEMPLATE   = "Q: {question}\\nA:"
    GROUNDED_TEMPLATE = "{evidence}\\n\\nQ: {question}\\nA:"

The proxy grounds chat-completion requests, so it adapts the grounded form to a
system message: the evidence block is prepended verbatim, with instruction
lines. This is an adaptation for chat clients, not the measured condition —
benchmark claims always refer to the completion templates above.

The closing warn-don't-suppress sentence is the answerability guard for
unanswerable/false-premise questions. Measured on the deployment instrument
(qwen3.5:4b-q4_K_M via llama.cpp, 150 unanswerable probes from the public
evalsets, chat shape): fabrication 0.080 -> 0.033 with the sentence, with
answerable-probe accuracy unchanged (0.300 both arms, n=150).
"""

PROMPT_TEMPLATE = "Q: {question}\nA:"
GROUNDED_TEMPLATE = "{evidence}\n\nQ: {question}\nA:"

SYSTEM_TEMPLATE = (
    "Reference facts from the Ballast knowledge corpus (level L{level}):\n\n"
    "{evidence}\n\n"
    "Use these facts when they are relevant to the user's question. "
    "If a fact above contradicts your memory, trust the fact. "
    "If the facts above do not contain the answer, say so plainly instead "
    "of guessing; related facts about an entity are not evidence that an "
    "answer exists."
)


def system_message(evidence_blocks: list[str], level: int) -> str:
    return SYSTEM_TEMPLATE.format(evidence="\n\n".join(evidence_blocks), level=level)
