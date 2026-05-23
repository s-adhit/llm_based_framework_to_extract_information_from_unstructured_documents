#prompt_builder.py
import json
import textwrap
from src.config import (
    PROMPTING_STRATEGY,
    FEW_SHOT_EXAMPLES,
    CATEGORIES,
    DEFINITIONS
)

_COT_STRATEGIES = {"zero_shot_cot", "few_shot_cot"}
_BASE_STRATEGY = {
    "zero_shot_cot": "zero_shot",
    "few_shot_cot":  "few_shot",
}

_COT_INSTRUCTION = """THINK STEP BY STEP before producing your answer:
  1. Apply the Subject Test to each sentence.
  2. Split clauses; check the PoI's role in each.
  3. Match each clause to a definition.
  4. Determine the final label(s).
Your final output must still be ONLY the JSON array with "id" and "c". No reasoning text."""


def format_examples(examples, title, include_reasoning=False):
    if not examples:
        return ""

    formatted = f"\n{title}:\n"
    for ex in examples:
        if include_reasoning:
            input_block  = json.dumps(ex.get('input',  []), ensure_ascii=False)
            output_block = json.dumps(ex.get('output', []), ensure_ascii=False)
            reasoning    = ex.get('reasoning', '')
            formatted += (
                f"Input:     {input_block}\n"
                f"Reasoning: {reasoning}\n"
                f"Output:    {output_block}\n---\n"
            )
        else:
            category = ex.get('category') or ex.get('c')
            text     = ex.get('text', "No text provided")
            formatted += f"Text: \"{text}\"\nCategory: {category}\n---\n"

    return formatted


def format_definitions(definitions_list):
    def_str = ""
    for d in definitions_list:
        for cat, desc in d.items():
            def_str += f"- {cat}: {desc}\n"
    return def_str


def build_batch_classification_prompt(batch_items, previous_sentence=None):
    """
    previous_sentence: the raw text of the sentence immediately before this
                       batch's first sentence. Used as context only — no label.
    """
    # 1. Extract input sentences
    input_data_for_llm = []
    for item in batch_items:
        content  = item.get('data', {})
        id_val   = content.get('sent_id')
        text_val = content.get('text', "Missing text")
        input_data_for_llm.append({"id": id_val, "text": text_val})

    # 2. Resolve effective strategy and CoT flag
    is_cot             = PROMPTING_STRATEGY in _COT_STRATEGIES
    effective_strategy = _BASE_STRATEGY.get(PROMPTING_STRATEGY, PROMPTING_STRATEGY)

    # 3. Build examples block
    examples_str = ""
    if effective_strategy == "few_shot":
        examples_str = format_examples(
            FEW_SHOT_EXAMPLES,
            "Reference Examples",
            include_reasoning=is_cot
        )

    # 4. Previous sentence context block (all modes)
    if previous_sentence:
        context_block = f"\nPREVIOUS SENTENCE (context only — do not classify):\n\"{previous_sentence}\"\n"
    else:
        context_block = ""

    # 5. CoT block (empty string when not active)
    cot_block = f"\n{_COT_INSTRUCTION}\n" if is_cot else ""

    # 6. Assemble prompt
    category_list_str = "\n".join([f"- {c}" for c in CATEGORIES])
    definitions_str   = format_definitions(DEFINITIONS)

    prompt = f"""You are a data annotation expert. Classify each sentence below.

        DECISION FLOW:
        1. Subject test: Remove the subject — does the sentence still make complete sense? If yes → ["Others"]. Stop.
        2. Clause split: Split on conjunctions/semicolons. For each clause, check if PoI is the active focus.
        3. Label each clause using definitions below. Surface-level only — no inferences.
        4. Aggregate: Include any category where ≥1 clause qualifies. ["Others"] only if ALL clauses are Others. Others is mutually exclusive.

        CATEGORIES & DEFINITIONS:
        {category_list_str}
        {definitions_str}
        {examples_str}{context_block}{cot_block}
        INPUT:
        {json.dumps(input_data_for_llm, indent=2, ensure_ascii=False)}

        OUTPUT (STRICT) — return ONLY a JSON array of objects:
        - "id": (copy the ID from the input)
        - "c": (a JSON list of strings; first element must be the most apparent category)

        Rules: Others cannot combine with any other label. No markdown. No explanation."""

    return textwrap.dedent(prompt).strip()