import json
from src.config import PROMPTING_STRATEGY, FEW_SHOT_EXAMPLES

def format_examples(examples, title):
    """Helper to format a list of dicts into a prompt string."""
    if not examples:
        return ""
        
    formatted = f"\n{title}:\n"
    for ex in examples:
        # Use .get('category') or .get('class') to handle both static config and dynamic memory
        category = ex.get('category') or ex.get('class')
        formatted += f"Text: \"{ex['text']}\"\nCategory: {category}\n---\n"
    return formatted

def build_batch_classification_prompt(batch_items,dynamic_memory=None):
    """
    Constructs the classification prompt using the user's specific template.
    """

    examples_str = ""
    instruction_line = "Classify based on the provided category definitions."

    if PROMPTING_STRATEGY == "few_shot":
        # Strategy 2: Static Few-Shot
        examples_str = format_examples(FEW_SHOT_EXAMPLES, "Reference Examples")
        instruction_line = "Classify based on the provided definitions and reference examples."

    elif PROMPTING_STRATEGY == "memory_prompt":
        # Strategy 3: Dynamic Memory Prompt
        if dynamic_memory:
            examples_str = format_examples(dynamic_memory, "Recent Annotations (Memory)")
            instruction_line = f"Classify based on definitions and the last {len(dynamic_memory)} annotations you successfully performed. Emulate the style and classification logic of these recent examples."
        else:
            instruction_line = "Classify based on definitions only, as the dynamic memory is currently empty."

    
    input_data = [{"id": item['temp_id'], "text": item['text']} for item in batch_items]
    input_json_str = json.dumps(input_data, ensure_ascii=False)

    prompt = f"""
    Role: Data Annotation Expert.

    Task:
    Classify each of the {len(batch_items)} sentences into exactly ONE category.
    {instruction_line}

    Categories:
    - Background
    - Personal Life
    - Achievements
    - Education
    - Work Experience
    - Interests
    - Others

    Category Definitions:
    - Background: Early-life context such as birthplace, childhood environment, family circumstances, parents’ occupations, socioeconomic or cultural upbringing.
    - Personal Life: Private, non-professional matters including relationships, marriage, divorce, children, lifestyle, or household details.
    - Achievements: Formal recognitions, awards, honors, records, or institutionally granted distinctions.
    - Education: Schooling, university attendance, academic degrees, certifications, or structured training programs.
    - Work Experience: Professional roles, jobs, duties, leadership positions, organizational affiliations, or career-related contributions.
    - Interests: Hobbies, leisure activities, artistic pursuits, sports, or passions outside professional obligations.
    - Others: Anything that does not fit the categories above, including controversies, legal matters, political incidents, health issues, financial problems, rumors, or miscellaneous facts.

    {examples_str}
    
    Input:
    {input_json_str}

    Output Format (strict):
    Return a JSON array ONLY.  
    Each element must be an object with EXACTLY these fields:
    {{
    "id": "<string or number, copied exactly from input>",
    "class": "<one of the category names>",
    "confidence_score": "<float between 0.0 and 1.0>"
    }}

    Constraints:
    - Do not output any text outside the JSON array.
    - Do not add fields, remove fields, or change field names.
    - Assign exactly ONE category per sentence.
    - Base classification strictly on the content of each individual sentence.
    """
    return prompt