import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in a .env file.")

# Model Configuration
MODEL_NAME = "gemini-2.5-flash"

# 200 items per batch = ~51 requests total for 10k items.
# This leaves you ~200 requests buffer for retries/testing.
BATCH_SIZE = 200

# Paths
INPUT_FILE_PATH = "data/input_data.json"
OUTPUT_FILE_PATH = "data/output_data.json"

# Options: "zero_shot" or "few_shot" or "memory_prompt"
PROMPTING_STRATEGY = "memory_prompt" 
MEMORY_SIZE = 15

FEW_SHOT_EXAMPLES = [
    {"text": "Harold Paul Freeman was born on March 2, 1933, in Washington, D.C. to Clyde and Lucille Thomas Freeman.", "category": "Background"},
    {"text": "The family name \"Freeman\" was chosen by his great-great-grandfather who bought himself free from slavery on a North Carolina plantation.", "category": "Background"},
    {"text": "He married in 2010 and has two kids.", "category": "Personal Life"},
    {"text": "In 1978, he was received a Distinguished Alumni Achievement Award from Catholic University.", "category": "Achievements"},
    {"text": "In 1992, he was inducted into the university's Athletes Hall of Fame.", "category": "Achievements"},
    {"text": "Harold completed his high school education at Dunbar High School, then an academically elite but segregated institution in Washington, D.C.", "category": "Education"},
    {"text": "He went on to study medicine at Howard University Medical School in Washington, D.C., also a historically Black school.", "category": "Education"},
    {"text": "She also served briefly as an Army nurse.", "category": "Work Experience"},
    {"text": "Bai's first major American film role was in The Crow (1994), where she played the half sister and lover of the main villain, Top Dollar.", "category": "Work Experience"},
    {"text": "He is an avid mountaineer.", "category": "Interests"},
    {"text": "The company faced a lawsuit.", "category": "Others"}
]

# Reverted to Title Case
CATEGORIES = [
    "Background",
    "Personal Life",
    "Achievements",
    "Education",
    "Work Experience",
    "Interests",
    "Others"
]