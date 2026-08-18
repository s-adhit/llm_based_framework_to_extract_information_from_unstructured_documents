import json
import os
import shutil
import csv
from collections import deque
from src.config import MEMORY_SIZE


class MemoryManager:
    """Manages a rolling buffer of the last N annotations."""
    def __init__(self, max_size=MEMORY_SIZE): 
        self.memory = deque(maxlen=max_size)

    def add_batch(self, batch_inputs, batch_results):
        """Adds all results from the batch directly to memory."""
        # Create a map for easy lookup: {id: result_object}
        result_map = {r.get('id'): r for r in batch_results if 'id' in r}

        for input_item in batch_inputs:
            t_id = input_item['temp_id']
            
            if t_id in result_map:
                prediction = result_map[t_id]
                # Store the classified text and the result
                self.memory.append({
                    "text": input_item['text'],
                    "category": prediction.get('class', "Others") 
                })

    def get_examples(self):
        """Returns the current list of memory items."""
        return list(self.memory)

def load_json(filepath):
    """Loads JSON data. Returns empty list if file missing."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {filepath} is not valid JSON.")
            return []

def save_json(data, filepath):
    """
    Safely saves JSON data. 
    Writes to a temp file first, then renames it to prevent corruption
    if the script stops mid-write.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    temp_path = filepath + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    shutil.move(temp_path, filepath)

def load_contrastive_examples(filepath):
    """
    Loads contrastive examples from CSV file.
    Converts conflict pairs into structured examples format for CRWiki strategy.
    
    Expected CSV columns:
    - conflict_pair: "Category1 ↔ Category2"
    - true_class: The correct category
    - confused_with: The category it might be confused with
    - text: The example sentence
    - rationale_belongs: Why it belongs to true_class
    - rationale_not_confused: Why it doesn't belong to confused_with
    """
    if not os.path.exists(filepath):
        print(f"Contrastive examples file not found: {filepath}")
        return []
    
    contrastive_examples = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('text') or not row.get('true_class'):
                    continue
                
                # Create a contrastive example that shows the difference
                example = {
                    "conflict_pair": row.get('conflict_pair', ''),
                    "true_class": row.get('true_class', ''),
                    "confused_with": row.get('confused_with', ''),
                    "text": row.get('text', ''),
                    "rationale_correct": row.get('rationale_belongs', ''),
                    "rationale_incorrect": row.get('rationale_not_confused', '')
                }
                contrastive_examples.append(example)
        
        print(f"Loaded {len(contrastive_examples)} contrastive examples from {filepath}")
        return contrastive_examples
    
    except Exception as e:
        print(f"Error loading contrastive examples from {filepath}: {e}")
        return []