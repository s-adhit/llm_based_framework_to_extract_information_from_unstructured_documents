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
PROMPTING_STRATEGY = "few_shot" 
MEMORY_SIZE = 15

FEW_SHOT_EXAMPLES = [
    {"input": [{"id": "s1", "text": "She was Professor of Physics at the Open University from 1991 to 2001."}],
     "output": [{"id": "s1", "c": ["Work Experience"]}],
     "reasoning": "Specific job title at a named institution with explicit tenure dates."},

    {"input": [{"id": "s2", "text": "Levin endorsed Orrin Hatch when Levin was being sponsored by Americans for Prosperity (AFP) which also endorsed Hatch."}],
     "output": [{"id": "s2", "c": ["Work Experience"]}],
     "reasoning": "Levin is the active agent performing a professional action (endorsement) tied to a formal sponsorship — a substantive output of his advocacy role."},

    {"input": [{"id": "s3", "text": "In 2019, she was awarded an Honorary Doctorate of Laws from the University of Bath."}],
     "output": [{"id": "s3", "c": ["Achievements"]}],
     "reasoning": "Formal award bestowed by an external institution."},

    {"input": [{"id": "s4", "text": "In 2020, she was included by the BBC in a list of seven important but little-known British female scientists."}],
     "output": [{"id": "s4", "c": ["Achievements"]}],
     "reasoning": "External organization formally recognized the individual's significance."},

    {"input": [{"id": "s5", "text": "There she was favourably impressed by her physics teacher, Mr. Tillott, and stated: You do not have to learn lots and lots ... of facts; you just learn a few key things, and ... then you can apply and build and develop from those ... He was a really good teacher and showed me, actually, how easy physics was."}],
     "output": [{"id": "s5", "c": ["Motivators"]}],
     "reasoning": "External influence (teacher) and its resulting effect on subject's perspective are both explicitly stated."},

    {"input": [{"id": "s6", "text": "Matsuo said that before Senko Riot, adults in her hometown would laugh at her when she told them she wanted to make a living in music, but the other acts she met there showed her it was not a pipe dream and she decided to move to Tokyo to make her dream a reality."}],
     "output": [{"id": "s6", "c": ["Motivators"]}],
     "reasoning": "External influence (other acts) caused a documented career decision (moving to Tokyo) — both influence and outcome are explicit."},

    {"input": [{"id": "s7", "text": "She also enjoyed her father's books on astronomy."}],
     "output": [{"id": "s7", "c": ["Interests"]}],
     "reasoning": "Recreational activity outside any formal or professional obligation."},

    {"input": [{"id": "s8", "text": "This saw her fall in love with classic rock, and she started listening to The Beatles and Led Zeppelin and watching Woodstock DVDs."}],
     "output": [{"id": "s8", "c": ["Interests"]}],
     "reasoning": "Leisure activities with no professional obligation or output."},

    {"input": [{"id": "s9", "text": "She grew up in Lurgan and attended the Preparatory Department of Lurgan College from 1948 to 1956."}],
     "output": [{"id": "s9", "c": ["Education"]}],
     "reasoning": "Named institution with enrollment period. 'Grew up' is Background but Education dominates."},

    {"input": [{"id": "s10", "text": "Raised in the Church of the Nazarene (which he ultimately left in 1968), he won a scholarship to the Church-affiliated Bethany Nazarene College (now Southern Nazarene University) in Bethany, Oklahoma, in 1954 and graduated with a B.A. in philosophy in 1958."}],
     "output": [{"id": "s10", "c": ["Education"]}],
     "reasoning": "Scholarship and degree at a named institution — both Education signals. Religious upbringing is contextual Background but Education dominates."},

    {"input": [{"id": "s11", "text": "Bell Burnell was born in Lurgan, County Armagh, Northern Ireland, to M. Allison and G. Philip Bell."}],
     "output": [{"id": "s11", "c": ["Background"]}],
     "reasoning": "Birth location and parental names — foundational Background facts."},

    {"input": [{"id": "s12", "text": "Hart was born in Ottawa, Kansas, the son of Nina (née Pritchard) and Carl Riley Hartpence, a farm equipment salesman."}],
     "output": [{"id": "s12", "c": ["Background"]}],
     "reasoning": "Birthplace, family composition, and parental occupation describing the socioeconomic context of childhood."},

    {"input": [{"id": "s13", "text": "She realized she could achieve this by starting a band; she could sing songs, design the album covers, and merchandise."}],
     "output": [{"id": "s13", "c": ["Learnings"]}],
     "reasoning": "Subject draws an explicit strategic conclusion framed as a personal insight, not an event description."},

    {"input": [{"id": "s14", "text": "He maintained that treating the shares of a company like baseball cards is a losing strategy because it requires one to predict the behavior of often irrational and emotional human beings."}],
     "output": [{"id": "s14", "c": ["Learnings"]}],
     "reasoning": "Subject states a concluded professional insight framed as a takeaway ('maintained that'), not an event."},

    {"input": [{"id": "s15", "text": "This race for the nomination was the most recent occasion that a major party's presidential nomination has gone all the way to the convention."}],
     "output": [{"id": "s15", "c": ["Others"]}],
     "reasoning": "Subject test: PoI absent — historical fact about an event, not about the individual."},

    {"input": [{"id": "s16", "text": "The Daily Telegraph science reporter shortened 'pulsating radio source' to pulsar."}],
     "output": [{"id": "s16", "c": ["Others"]}],
     "reasoning": "Third-party action with PoI entirely absent — passes subject test → Others."},
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