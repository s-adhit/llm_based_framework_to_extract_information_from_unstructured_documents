# gemini_client.py
from google import genai
from google.genai import types
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ClientError, ServerError
from src.config import API_KEY, MODEL_NAME, CATEGORIES

client = genai.Client(api_key=API_KEY)

# OPTIMIZED SCHEMA (Multi-label, Surface-level)
# 'c' is now an ARRAY to support one or more labels.
# 's' (confidence) has been removed as requested.
RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "id": {"type": "INTEGER"},
            "c": {
                "type": "ARRAY", 
                "items": {
                    "type": "STRING", 
                    "enum": CATEGORIES
                },
                "minItems": 1 # Ensures at least one label is always returned
            }
        },
        "required": ["id", "c"]
    }
}

@retry(
    retry=retry_if_exception_type((ClientError, ServerError)),
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=4, min=15, max=60)
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
        
        # The SDK's 'parsed' attribute will now return a list of strings for 'c'
        if hasattr(response, 'parsed') and response.parsed is not None:
             return response.parsed
        
        return json.loads(response.text)

    except Exception as e:
        print(f"Batch Error: {str(e)[:100]}...")
        if "429" in str(e):
            print("Rate Limit Hit! Creating a long pause...")
        raise e