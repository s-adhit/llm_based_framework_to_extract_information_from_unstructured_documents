# gemini_client.py
from google import genai
from google.genai import types
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError, ServerError
from src.config import API_KEY, MODEL_NAME, CATEGORIES

client = genai.Client(api_key=API_KEY)

# OPTIMIZED SCHEMA (Token Saving)
# We use 'c' for class and 's' for score to save output tokens.
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "INTEGER"},
            "c": {"type": "STRING", "enum": CATEGORIES}, # Abbreviated 'class'
            "s": {"type": "NUMBER"}                       # Abbreviated 'confidence_score'
        },
        "required": ["id", "c", "s"]
    }
}

@retry(
    retry=retry_if_exception_type((ClientError, ServerError)),
    stop=stop_after_attempt(5), # Reduce attempts to save quota
    wait=wait_exponential(multiplier=4, min=15, max=60) # Increase wait time
)
def get_batch_classification(batch_items, prompt):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA
            )
        )
        
        if hasattr(response, 'parsed') and response.parsed is not None:
             return response.parsed
        
        return json.loads(response.text)

    except Exception as e:
        print(f"Batch Error: {str(e)[:100]}...")
        # If we hit a 429 (Rate Limit), we must ensure we don't retry too fast
        if "429" in str(e):
            print("Rate Limit Hit! Creating a long pause...")
        raise e