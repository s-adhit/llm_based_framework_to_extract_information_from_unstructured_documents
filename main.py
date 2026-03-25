#main.py
import os
import time
from tqdm import tqdm
from src.config import BATCH_SIZE, PROMPTING_STRATEGY, INPUT_FILE_PATH, OUTPUT_FILE_PATH
from src.utils import load_json, save_json, MemoryManager
from src.prompt_builder import build_batch_classification_prompt
from src.gemini_client import get_batch_classification

def chunk_data(data, size):
    """Yield successive n-sized chunks from data."""
    for i in range(0, len(data), size):
        yield data[i:i + size]

def initialize_output_file():
    """Ensures output file exists, otherwise clones input."""
    if os.path.exists(OUTPUT_FILE_PATH):
        print(f"Resuming from existing output file: {OUTPUT_FILE_PATH}")
        return load_json(OUTPUT_FILE_PATH)
    else:
        print(f"No output file found. Creating {OUTPUT_FILE_PATH} from input...")
        data = load_json(INPUT_FILE_PATH)
        if not data:
            raise ValueError(f"Input file {INPUT_FILE_PATH} is empty or missing!")
        save_json(data, OUTPUT_FILE_PATH)
        return data

def main():
    # 1. Load Data
    data = initialize_output_file()
    
    # 2. Identify Pending Items
    pending_items = []
    for i, item in enumerate(data):
        inner_data = item.get('data', {})
        category = inner_data.get('category')
        
        # Only add to pending if 'category' is missing, "API_Missed", or "Batch_Error"
        # This allows the script to RETRY items that failed previously
        if not category or category == ["API_Missed"] or category == ["Batch_Error"]:
            item['temp_id'] = i  
            pending_items.append(item)
    
    print(f"Strategy: {PROMPTING_STRATEGY.upper()} | Pending: {len(pending_items)}")
    
    if not pending_items:
        print("All items are already classified. Done!")
        return

    # Initialize Memory ONLY if strategy is memory_prompt
    memory = MemoryManager() if PROMPTING_STRATEGY == "memory_prompt" else None

    # 3. Batch Processing
    chunks = list(chunk_data(pending_items, BATCH_SIZE))
    
    for batch_idx, batch in tqdm(enumerate(chunks), total=len(chunks), desc="Processing"):
        try:
            # --- CONTEXT RETRIEVAL & API CALL ---
            dynamic_context = memory.get_examples() if memory else None
            prompt = build_batch_classification_prompt(batch, dynamic_memory=dynamic_context)
            
            # --- API CALL ---
            results = get_batch_classification(batch, prompt)
            
            # Map results using STRING keys (e.g., {"0": {...}})
            result_map = {str(r.get('id')): r for r in results if r.get('id') is not None}
            
            # --- RESULTS PROCESSING ---
            missed_in_this_batch = 0
            for item in batch:
                t_idx = item['temp_id']
                
                # FIX: Extract from data -> sent_id
                inner_data = item.get('data', {})
                raw_id = inner_data.get('sent_id')
                
                # Force to string so it matches the result_map keys
                sent_id_str = str(raw_id) 
                
                if sent_id_str in result_map:
                    res = result_map[sent_id_str]
                    # This adds 'category' inside the 'data' dict of your object
                    item['data']['category'] = res.get('c', ["Others"])
                else:
                    item['data']['category'] = ["API_Missed"]
                    missed_in_this_batch += 1

            # Success Tracker
            if missed_in_this_batch > 0:
                print(f"\n[!] Batch {batch_idx}: {missed_in_this_batch} missed. Looked for '{sent_id_str}'")
            else:
                print(f"\n[✓] Batch {batch_idx}: 100% Match.")

            # --- MEMORY UPDATE ---
            if memory:
                memory.add_batch(batch, results)

            # --- IMMEDIATE SAVE (Checkpointing) ---
            save_json(data, OUTPUT_FILE_PATH)

        except Exception as e:
            print(f"\nBatch {batch_idx} failed: {e}")
            # Mark errors so the next run knows these failed
            for item in batch:
                if 'data' in data[item['temp_id']]:
                    data[item['temp_id']]['data']['category'] = ["Batch_Error"]
            
            save_json(data, OUTPUT_FILE_PATH)

        # Rate Limit Buffer
        time.sleep(1.0)

    print(f"Pipeline complete! Results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()