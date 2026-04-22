"""
Minimal prompts for the deriver module optimized for speed.

This module contains simplified prompt templates focused only on observation extraction.
NO peer card instructions, NO working representation - just extract observations.
"""

from functools import cache
from inspect import cleandoc as c

from src.utils.tokens import estimate_tokens


def minimal_deriver_prompt(
    peer_id: str,
    messages: str,
) -> str:
    """
    Generate minimal prompt for fast observation extraction.

    Args:
        peer_id: The ID of the user being analyzed.
        messages: All messages in the range (interleaving messages and new turns combined).

    Returns:
        Formatted prompt string for observation extraction.
    """
    return c(
        f"""
You extract explicit observations for durable memory.

Return JSON matching the schema exactly.
- Always return an object with key "explicit".
- "explicit" must be an array.
- Each item must be an object with a single key: "content".
- Never return null.
- Never return markdown, explanations, or prose outside JSON.

Task:
Extract explicit, durable, self-contained facts about {peer_id} from THEIR messages only.
Use other speakers only as context for resolving references.

Include only facts that are:
- directly stated or very tightly paraphrased from {peer_id}'s own words
- useful as durable memory or preferences
- understandable on their own without the source transcript

Do not include:
- generic chatter, acknowledgements, or task boilerplate
- facts about other people unless the fact is specifically about {peer_id}'s relationship or preference
- speculative inferences

If there are no durable explicit facts, return:
{{"explicit":[]}}

Good examples:
- "I prefer technical collaboration in Vietnamese." -> {{"content":"{peer_id} prefers technical collaboration in Vietnamese."}}
- "I want runtime-verified evidence during production debugging." -> {{"content":"{peer_id} wants runtime-verified evidence during production debugging."}}
- "I prefer concise answers." -> {{"content":"{peer_id} prefers concise answers."}}

Messages to analyze:
<messages>
{messages}
</messages>
"""
    )


@cache
def estimate_minimal_deriver_prompt_tokens() -> int:
    """Estimate base prompt tokens (cached)."""
    try:
        prompt = minimal_deriver_prompt(
            peer_id="",
            messages="",
        )
        return estimate_tokens(prompt)
    except Exception:
        return 300
