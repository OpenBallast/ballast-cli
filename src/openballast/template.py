"""Prompt templates.

The research-measured conditions (logprob scoring) are completion-style:

    PROMPT_TEMPLATE   = "Q: {question}\\nA:"
    GROUNDED_TEMPLATE = "{evidence}\\n\\nQ: {question}\\nA:"

The proxy grounds chat-completion requests, so it adapts the grounded form to a
system message: the evidence block is prepended verbatim, with two instruction
lines. This is an adaptation for chat clients, not the measured condition —
benchmark claims always refer to the completion templates above.
"""

PROMPT_TEMPLATE = "Q: {question}\nA:"
GROUNDED_TEMPLATE = "{evidence}\n\nQ: {question}\nA:"

SYSTEM_TEMPLATE = (
    "Reference facts from the Ballast knowledge corpus (level L{level}):\n\n"
    "{evidence}\n\n"
    "Use these facts when they are relevant to the user's question. "
    "If a fact above contradicts your memory, trust the fact."
)


def system_message(evidence_blocks: list[str], level: int) -> str:
    return SYSTEM_TEMPLATE.format(evidence="\n\n".join(evidence_blocks), level=level)
