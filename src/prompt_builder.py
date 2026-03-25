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

    # 4. Construct Final Prompt with your specific conditions
    prompt = f"""
        Role: Data Annotation Expert.

        Task:
        Classify each of the {len(batch_items)} sentences based on the following specific logical flow:
        
        1. **Active Agent Check:** First, determine if the subject is the **active agent** in the sentence. If the subject is NOT the active agent (e.g., they are a passive recipient or bystander), the sentence MUST be marked as ["Others"].
        2. **Clause Breakdown:** If the subject IS the active agent, break the sentence down into its constituent clauses.
        3. **Clause Classification:** For each clause, select the appropriate category based ONLY on surface-level information (no implications).
        4. **Aggregation:** - If ANY clause qualifies for a specific category (Background, Achievements, etc.), include that category in the result list.
           - Mark the sentence as ["Others"] ONLY if ALL clauses in the sentence belong to the "Others" category.

        Categories:
        {category_list_str}

        Category Definitions:
        {definitions_str}
        {examples_str}
        
        Input JSON:
        {json.dumps(input_data_for_llm, indent=2, ensure_ascii=False)}

        Output Format (STRICT):
        Return a JSON array of objects with:
        - "id": (copy the ID from the input)
        - "c": (a JSON list of strings. The first element must be the most apparent category)

        Constraints:
        - Return ONLY the JSON array.
        - **Surface-Level Only:** Do not use internal logic or implications. If the text doesn't explicitly state it, don't label it.
        - **Exclusivity Rule:** The "Others" category is mutually exclusive. If a sentence is "Others", the list MUST contain ONLY ["Others"].
        - Do not include tables or formatting in your classification logic.
        """
    
    return textwrap.dedent(prompt).strip()