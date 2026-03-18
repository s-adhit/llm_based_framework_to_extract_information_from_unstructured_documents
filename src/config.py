import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in a .env file.")

# Model Configuration
MODEL_NAME = "gemini-3-flash-preview"

# 200 items per batch = ~51 requests total for 10k items.
# This leaves you ~200 requests buffer for retries/testing.
BATCH_SIZE = 100

# Paths
INPUT_FILE_PATH = "data/input_data.json"
OUTPUT_FILE_PATH = "data/output_data.json"

# Options: "zero_shot" or "few_shot" or "memory_prompt"
PROMPTING_STRATEGY = "zero_shot" 
MEMORY_SIZE = 15

FEW_SHOT_EXAMPLES = [
    {"text": "Paglia was born in Endicott, New York, the eldest child of Lydia Anne (ne Colapietro) and Pasquale Paglia.", "category": "Background"},
    {"text": "During her stays at a summer Girl Scout camp in Thendara, New York, she took on a variety of new names, including Anastasia (her confirmation name, inspired by the film Anastasia), Stacy, and Stanley.", "category": "Personal Life"},
    {"text": "In 2005, Paglia was named as one of the top 100 public intellectuals by the journals Foreign Policy and Prospect.", "category": "Achievements"},
    {"text": "She attended the Edward Smith Elementary School, T. Aaron Levy Junior High, and Nottingham Senior High School.", "category": "Education"},
    {"text": "In September 1976, she gave a public lecture drawing on that dissertation, in which she discussed Edmund Spenser's The Faerie Queene, followed by remarks on Diana Ross, Gracie Allen, Yul Brynner, and Stphane Audran.", "category": "Work Experience"},
    {"text": "She has expressed Interests in astrology and has written about it in several of her works, including Sexual Personae: \"I'm an astrologer  people don\'t mention this!", "category": "Interests"},
    {"text": "Rollyson and Paddock note that Sontag \"had her lawyer put our publisher on notice\" when she realized she was to be the subject.", "category": "Others"}
]

# Reverted to Title Case
CATEGORIES = [
    "Background",
    "Achievements",
    "Education",
    "Work Experience",
    "Interests",
    "Motivators",
    "Reflections",
    "Others",
]

DEFINITIONS = [
    {"Background":"This encompasses the individual's early socioeconomic circumstances (pre-adulthood), including parental occupations, family composition in childhood, living conditions during the formative years, and geographical location."},
    {"Achievements":"This category documents individuals' accomplishments, including professional recognition and awards."},
    {"Education":"This includes details of an individual's academic history, such as institutions attended, key educational decisions, and academic performance metrics (e.g., grades)."},
    {"Work Experience":"This category encompasses details of an individual's professional history, including positions held, employment durations in each role, and the contributions, efforts, decisions, and impact of their work."},
    {"Interests":"This category outlines individuals' recreational pursuits, proficiencies, and general interests."},
    {"Motivators":"This category records external influences, such as support, inspiration, and environmental or interpersonal pressure, that affect the individual."},
    {"Reflections":"This captures the ideological stances, personal philosophies, observations, and insights an individual has accumulated over their lifetime."},
    {"Others":"All remaining information not classifiable under the aforementioned categories."},
]