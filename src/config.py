#config.py
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
BATCH_SIZE = 30

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
    "Learnings",
    "Others",
]

DEFINITIONS = [
    {"Background":"Birth/origin, family composition, parental occupations, socioeconomic or material conditions of childhood (up to age 18). Siblings/extended family only when they shaped material circumstances. Does Not Include - Subjective feelings about upbringing; adult conditions; educational details."},
    {"Achievements":"An external party formally recognised the subject (award, prize, title), or the subject set a significant milestone, or did something demonstrably for the first time. Does Not Include –   General reputation; self-assessed success; normal job outputs without a specific external award; subjective praise or ovations."},
    {"Education":"Formal institutional learning: institution attended, degree/qualification pursued, academic performance (grades, scholarships). Vocational training in a formal institution qualifies. Does Not Include – Informal self-directed learning; non-accredited employer development."},
    {"Work Experience":"Subject performed a professional action or produced a work output as part of a role (formal or semi-formal). Includes quantitative performance stats (e.g. points per game, sales figures). Does not include – Passive roles; industry context; others' actions; subject is a bystander."},
    {"Interests":"Activity or skill pursued for personal enjoyment or self-development, outside professional obligations. Unpaid overlap with profession counts here. Does Not Include — Employer-required training; primary income sources; formal job certifications."},
    {"Motivators":"An external influence AND a documented change in the subject's thinking or career direction; both must appear explicitly (cause + effect required). Does Not Include - Ongoing situations with no documented change; internal values without external triggers; influences that produced no documented shift."},
    {"Learnings":" Subject draws a generalized professional insight, strategy principle, or lesson from their career. May reference others' behavior if the takeaway is about strategy — not a personal judgment of the person. Does Not Include - Personal judgments or opinions about specific named individuals or entities."},
    {"Others":"Purely contextual: world history, org descriptions, residence, family facts — subject is a passive bystander and no other label applies. Does Not Include - Cannot be combined with any other class. Use only when nothing else fits."},
]