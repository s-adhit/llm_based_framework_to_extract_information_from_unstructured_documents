#config.py
import os
import csv
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
INPUT_FILE_PATH = "data/llm_sentences.json"
OUTPUT_FILE_PATH = "data/crwiki_on_zero_shot_output.json"

# Options: "zero_shot" | "few_shot" | "zero_shot_cot" | "few_shot_cot" | "crwiki"
# crwiki: Contrastive Rationale Wiki (uses contrastive examples from conflicting_pairs_with_rationales.csv)
PROMPTING_STRATEGY = "crwiki"
MEMORY_SIZE = 15

FEW_SHOT_EXAMPLES = [
    {"input": [{"id": "s1", "text": "Raised in the Church of the Nazarene (which he ultimately left in 1968), he won a scholarship to the Church-affiliated Bethany Nazarene College (now Southern Nazarene University) in Bethany, Oklahoma, in 1954 and graduated with a B.A. in philosophy in 1958. "}],
     "output": [{"id": "s1", "c": ["Background", "Education"]}],
     "reasoning": "The sentence mentions the PoI, passing the subject test. It contains three key details: his upbringing in the Church of the Nazarene (Background), a scholarship to Southern Nazarene University (Education), and a B.A. in philosophy (Education). Therefore, the final labels are Background and Education."},

    {"input": [{"id": "s2", "text": "The music video for the song was released on December 11, 2012."}],
     "output": [{"id": "s2", "c": ["Others"]}],
     "reasoning": "Since the sentence does not mention PoI at all, it fails the subject test and is therefore labeled as Others."},

    {"input": [{"id": "s3", "text": "On February 27, 2022, he unilaterally terminated the contract with Zenit, citing Russia's aggression on Ukraine as the reason."}],
     "output": [{"id": "s3", "c": ["Work Experience", "Motivators"]}],
     "reasoning": "The sentence mentions the PoI, passing the subject test. It contains two pieces of information: he actively terminates his contract with Zenit (Work Experience) and does so due to Russia's aggression on Ukraine (Motivators). Therefore, the final labels are Work Experience and Motivators."},
     

    {"input": [{"id": "s5", "text": "Ma's initial practical training in advertising fostered in her \"a production and design aesthetic in the sense that she is responsive to the qualities and needs of materials as well as to the demands of place and public,\" while remaining critical of market and client."}],
     "output": [{"id": "s5", "c": ["Learnings"]}],
     "reasoning": "The sentence mentions the PoI (Ma), passing the subject test. It explains how her initial training in advertising shaped her design aesthetic, responsiveness to materials, and critical perspective on the market. Because this focuses entirely on the skills, perspectives, and insights she acquired during her training, it is labeled Learnings."},

    {"input": [{"id": "s6", "text": "He has frequently worked with filmmaker Wes Anderson, with whom he has shared writing and acting credits on the films Bottle Rocket (1996), Rushmore (1998), and The Royal Tenenbaums (2001) ”the latter received a nomination for the Academy Award and BAFTA Award for Best Screenplay."}],
     "output": [{"id": "s6", "c": ["Work Experience", "Achievements"]}],
     "reasoning": "The sentence mentions the PoI, passing the subject test. It highlights two main components: his extensive collaborative filmmaking, writing, and acting credits with Wes Anderson (Work Experience), and the subsequent Academy Award and BAFTA nominations for Best Screenplay (Achievements). Therefore, the final labels are Work Experience and Achievements."},

    {"input": [{"id": "s7", "text": "After moving from Beijing to Oklahoma, Ma started drawing and painting as an alternative to literature, from which she felt alienated having to speak a second language."}],
     "output": [{"id": "s7", "c": ["Interests"]}],
     "reasoning": "The sentence mentions the PoI (Ma), passing the subject test. It notes that after moving to Oklahoma, she picked up drawing and painting as a creative outlet due to feeling alienated from literature in a second language. Since this focuses on her pursuing these creative activities, it is labeled Interests."},

]

# Reverted to Title Case (8-Class Default)
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

# Merged 5-Class Taxonomy (For fair evaluation with baseline classifiers)
CATEGORIES_5CLASS = [
    "Background",
    "Achievements",
    "Education",
    "Work Experience",
    "Others",
]

DEFINITIONS_5CLASS = [
    {"Background":"Birth/origin, family composition, parental occupations, socioeconomic or material conditions of childhood (up to age 18). Siblings/extended family only when they shaped material circumstances. Does Not Include - Subjective feelings about upbringing; adult conditions; educational details."},
    {"Achievements":"An external party formally recognised the subject (award, prize, title), or the subject set a significant milestone, or did something demonstrably for the first time. Does Not Include –   General reputation; self-assessed success; normal job outputs without a specific external award; subjective praise or ovations."},
    {"Education":"Formal institutional learning: institution attended, degree/qualification pursued, academic performance (grades, scholarships). Vocational training in a formal institution qualifies. Does Not Include – Informal self-directed learning; non-accredited employer development."},
    {"Work Experience":"Subject performed a professional action or produced a work output as part of a role (formal or semi-formal). Includes quantitative performance stats (e.g. points per game, sales figures). Does not include – Passive roles; industry context; others' actions; subject is a bystander."},
    {"Others":"Purely contextual: world history, org descriptions, residence, family facts, personal interests, motivators, or general learnings where no specific target label applies. Does Not Include - Cannot be combined with any other class. Use only when nothing else fits."},
]

FEW_SHOT_EXAMPLES_5CLASS = [
    {"input": [{"id": "s1", "text": "Raised in the Church of the Nazarene (which he ultimately left in 1968), he won a scholarship to the Church-affiliated Bethany Nazarene College (now Southern Nazarene University) in Bethany, Oklahoma, in 1954 and graduated with a B.A. in philosophy in 1958. "}],
     "output": [{"id": "s1", "c": ["Background", "Education"]}],
     "reasoning": "The sentence mentions the PoI, passing the subject test. It contains three key details: his upbringing in the Church of the Nazarene (Background), a scholarship to Southern Nazarene University (Education), and a B.A. in philosophy (Education). Therefore, the final labels are Background and Education."},

    {"input": [{"id": "s2", "text": "The music video for the song was released on December 11, 2012."}],
     "output": [{"id": "s2", "c": ["Others"]}],
     "reasoning": "Since the sentence does not mention PoI at all, it fails the subject test and is therefore labeled as Others."},

    {"input": [{"id": "s3", "text": "On February 27, 2022, he unilaterally terminated the contract with Zenit, citing Russia's aggression on Ukraine as the reason."}],
     "output": [{"id": "s3", "c": ["Work Experience"]}],
     "reasoning": "The sentence mentions the PoI, passing the subject test. He unilaterally terminated his contract with Zenit (Work Experience) due to Russia's aggression on Ukraine. Therefore, the final label is Work Experience."},
     

    {"input": [{"id": "s5", "text": "Ma's initial practical training in advertising fostered in her \"a production and design aesthetic in the sense that she is responsive to the qualities and needs of materials as well as to the demands of place and public,\" while remaining critical of market and client."}],
     "output": [{"id": "s5", "c": ["Others"]}],
     "reasoning": "The sentence mentions the PoI (Ma), passing the subject test. It explains how her initial training in advertising shaped her design aesthetic, responsiveness to materials, and critical perspective on the market but doesn't fit any relevant labels. Therefore, it is classified as Others."},

    {"input": [{"id": "s6", "text": "He has frequently worked with filmmaker Wes Anderson, with whom he has shared writing and acting credits on the films Bottle Rocket (1996), Rushmore (1998), and The Royal Tenenbaums (2001) ”the latter received a nomination for the Academy Award and BAFTA Award for Best Screenplay."}],
     "output": [{"id": "s6", "c": ["Work Experience", "Achievements"]}],
     "reasoning": "The sentence mentions the PoI, passing the subject test. It highlights two main components: his extensive collaborative filmmaking, writing, and acting credits with Wes Anderson (Work Experience), and the subsequent Academy Award and BAFTA nominations for Best Screenplay (Achievements). Therefore, the final labels are Work Experience and Achievements."},

]

def _load_contrastive_examples():
    """
    Loads contrastive examples from CSV file for CRWiki strategy.
    Returns a list of contrastive example dictionaries.
    """
    csv_path = "data/conflicting_pairs_with_rationales.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: Contrastive examples file not found: {csv_path}")
        return []
    
    examples = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('text') or not row.get('true_class'):
                    continue
                
                example = {
                    "conflict_pair": row.get('conflict_pair', ''),
                    "true_class": row.get('true_class', ''),
                    "confused_with": row.get('confused_with', ''),
                    "text": row.get('text', ''),
                    "rationale_correct": row.get('rationale_belongs', ''),
                    "rationale_incorrect": row.get('rationale_not_confused', '')
                }
                examples.append(example)
        
        print(f"Loaded {len(examples)} contrastive examples from {csv_path}")
        return examples
    
    except Exception as e:
        print(f"Error loading contrastive examples: {e}")
        return []

# Populate contrastive examples at config load time
CONTRASTIVE_EXAMPLES = _load_contrastive_examples()