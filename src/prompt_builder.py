#prompt_builder.py
import json
import textwrap
from src.config import (
    PROMPTING_STRATEGY, 
    FEW_SHOT_EXAMPLES, 
    CATEGORIES, 
    DEFINITIONS
)

def format_examples(examples, title):
    """Helper to format examples into a clear string for the LLM."""
    if not examples:
        return ""
        
    formatted = f"\n{title}:\n"
    for ex in examples:
        # Check 'category' (static config) or 'c' (dynamic memory/schema)
        category = ex.get('category') or ex.get('c')
        text = ex.get('text', "No text provided")
        formatted += f"Text: \"{text}\"\nCategory: {category}\n---\n"
    return formatted

def format_definitions(definitions_list):
    """Converts the list of dicts from config into a clean bulleted list."""
    def_str = ""
    for d in definitions_list:
        for cat, desc in d.items():
            def_str += f"- {cat}: {desc}\n"
    return def_str

def build_batch_classification_prompt(batch_items, dynamic_memory=None):
    """
    Constructs the classification prompt using the nested data structure.
    """

    # 1. Extract data from the nested JSON structure
    # Expected structure: item['data']['text'] and item['data']['meta']['sent_id']
    input_data_for_llm = []
    for item in batch_items:
        # Get the 'data' block
        content = item.get('data', {})
        
        # FIX: Extract sent_id directly from 'data' (no 'meta')
        id_val = content.get('sent_id')
        text_val = content.get('text', "Missing text")

        input_data_for_llm.append({
            "id": id_val,
            "text": text_val
        })

    # 2. Determine Strategy Instructions
    examples_str = ""
    instruction_line = "Classify based on the provided category definitions."

    if PROMPTING_STRATEGY == "few_shot":
        examples_str = format_examples(FEW_SHOT_EXAMPLES, "Reference Examples")
        instruction_line = "Classify based on the provided definitions and reference examples."

    elif PROMPTING_STRATEGY == "memory_prompt" and dynamic_memory:
        examples_str = format_examples(dynamic_memory, "Recent Annotations (Memory)")
        instruction_line = (
            f"Classify based on definitions and the last {len(dynamic_memory)} "
            "successful annotations. Emulate the logic of these recent examples."
        )

    # 3. Format Categories and Definitions from Config
    category_list_str = "\n".join([f"- {c}" for c in CATEGORIES])
    definitions_str = format_definitions(DEFINITIONS)

    prompt = f"""You are a data annotation expert. Classify each sentence below.

        DECISION FLOW:
        1. Subject test: Remove the subject — does the sentence still make complete sense? If yes → ["Others"]. Stop.
        2. Clause split: Split on conjunctions/semicolons. For each clause, check if PoI is the active focus.
        3. Label each clause using definitions below. Surface-level only — no inferences.
        4. Aggregate: Include any category where ≥1 clause qualifies. ["Others"] only if ALL clauses are Others. Others is mutually exclusive.

        CATEGORIES & DEFINITIONS:
        {category_list_str}
        {definitions_str}
        {examples_str}

        INPUT:
        {json.dumps(input_data_for_llm, indent=2, ensure_ascii=False)}

        OUTPUT (STRICT) — return ONLY a JSON array of objects:
        - "id": (copy the ID from the input)
        - "c": (a JSON list of strings; first element must be the most apparent category)

        Rules: Others cannot combine with any other label. No markdown. No explanation."""
    
    return textwrap.dedent(prompt).strip()