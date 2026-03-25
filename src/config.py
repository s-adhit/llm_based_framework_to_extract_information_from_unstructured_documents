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
    {"Background":"This captures the individual's place of birth and upbringing, family composition, parental occupations, and socioeconomic status, and the material conditions of childhood (considered till age 18). Information about siblings or extended family qualifies only when it directly shaped the subject's material circumstances or opportunities."},
    {"Achievements":"This captures individuals' accomplishments, including professional recognition, honours, and awards. Achievements can also take the form of being the first to do something significant. The recognition must originate from outside the subject and be discrete."},
    {"Education":"This includes details about an individual's academic history, such as institutions attended, significant educational choices (e.g., which degree to pursue), and academic performance metrics (e.g., grades and GPA). Vocational or professional training qualifies when it takes place within a formal institutional setting."},
    {"Work Experience":"This captures a person's professional details, including job titles, tenure in each role, and key actions. It highlights the main outcomes or products of each position, such as laws drafted, performances delivered, music released, or campaigns led, provided they are central to the job's purpose. The subject must be the active agent. Includes professional decisions, outputs, and actions taken in a formal or semi-formal occupational capacity, even when no job title or duration is explicitly stated. Quantitative performance metrics (e.g., sales figures, sports statistics, publication counts) are also considered professional outputs."},
    {"Interests":"This captures activities, pursuits, and skills the subject engaged with for personal pleasure or development, clearly within their leisure life-space and independent of professional obligations. Includes recreational habits, informal skill-building, cultural consumption, and creative pursuits. When a subject's hobby overlaps with their profession, the sentence qualifies here only if it describes the activity in a non-professional, non-compensated context."},
    {"Motivators":"This captures external influences, such as mentors, peers, or major events that directly cause a documented change in someone's thinking or career, such as rethinking a field or starting something new. Internal motivations, unchanging situations, or minor interests that don’t lead to change are not included.  Both the external influence and its effect on the subject must be present in the sentence or its immediate context."},
    {"Learnings":"This captures the professional insights and lessons an individual has acquired over their career. This category does not contain observations & judgements about other people. However, a sentence qualifies even if the insight references the behaviour of others, as long as the takeaway is about strategy, approach, or understanding rather than a personal judgment of a specific named individual."},
    {"Others":"All remaining information not classifiable under the aforementioned categories. It typically captures information where the subject is a passive recipient of circumstances or a bystander to external events. Includes world history, descriptions of organizations, and personal life facts (like residence or family) that do not involve the subject making an active professional or intellectual choice. This is an exclusive class, i.e., you can not mark a sentence as both Others and a different class. Only choose this class if the sentence cannot be labeled in the above classes."},
]