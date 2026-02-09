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
        content = item.get('data', {})
        text_val = content.get('text', "Missing text")
        meta = content.get('meta', {})
        
        # We use 'sent_id' as the ID so main.py can map it back easily
        id_val = meta.get('sent_id', item.get('temp_id'))
        
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

    # 4. Construct Final Prompt
    prompt = f"""
    Role: Data Annotation Expert.

    Task:
    Classify each of the {len(batch_items)} sentences into exactly ONE category.
    {instruction_line}

    Categories:
    {category_list_str}

    Category Definitions:
    {definitions_str}
    {examples_str}
    
    Input JSON:
    {json.dumps(input_data_for_llm, indent=2, ensure_ascii=False)}

    Output Format (STRICT):
    Return a JSON array of objects with these EXACT keys:
    - "id": (copy the ID from the input)
    - "c": (the chosen category name)
    - "s": (confidence score as a float 0.0-1.0)

    Constraints:
    - Return ONLY the JSON array. No preamble or conversational filler.
    - Base classification strictly on the provided definitions.
    """
    
    return textwrap.dedent(prompt).strip()