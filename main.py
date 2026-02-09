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
    # We check inside the 'data' block for the 'class' key to decide if it's done.
    pending_items = []
    for i, item in enumerate(data):
        inner_data = item.get('data', {})
        if "class" not in inner_data:
            item['temp_id'] = i  # Store index for easy mapping back
            pending_items.append(item)
    
    print(f"Strategy: {PROMPTING_STRATEGY.upper()} | Pending: {len(pending_items)}")
    
    if not pending_items:
        print("All items are already classified. Done!")
        return

    # Initialize Memory ONLY if strategy is memory_prompt
    memory = MemoryManager() if PROMPTING_STRATEGY == "memory_prompt" else None

    # 3. Batch Processing
    chunks = list(chunk_data(pending_items, BATCH_SIZE))
    
    for batch_idx, batch in tqdm(enumerate(chunks), total=len(chunks), desc="Processing Batches"):
        try:
            # --- CONTEXT RETRIEVAL ---
            dynamic_context = memory.get_examples() if memory else None
            prompt = build_batch_classification_prompt(batch, dynamic_memory=dynamic_context)
            
            # --- API CALL ---
            # returns list of dicts: [{"id": 0, "c": "Category", "s": 0.95}, ...]
            results = get_batch_classification(batch, prompt)
            
            # Create a map using string IDs for robust matching
            result_map = {str(r.get('id')): r for r in results if 'id' in r}
            
            # --- RESULTS PROCESSING ---
            for item in batch:
                t_idx = item['temp_id']
                # Use the same 'sent_id' used in the prompt_builder
                sent_id = str(item.get('data', {}).get('meta', {}).get('sent_id'))
                
                if sent_id in result_map:
                    res = result_map[sent_id]
                    # Assigning INSIDE the 'data' block as requested
                    data[t_idx]['data']['class'] = res.get('c', "Others")
                    data[t_idx]['data']['confidence_score'] = res.get('s', 0.0)
                else:
                    data[t_idx]['data']['class'] = "API_Missed"
                    data[t_idx]['data']['confidence_score'] = 0.0

            # --- MEMORY UPDATE (Feedback Loop) ---
            if memory:
                formatted_results = []
                for item in batch:
                    t_idx = item['temp_id']
                    formatted_results.append({
                        "text": item.get('data', {}).get('text'),
                        "c": data[t_idx]['data']['class']
                    })
                memory.add_batch(formatted_results)

        except Exception as e:
            print(f"\nBatch {batch_idx} failed: {e}")
            # Mark as error inside the 'data' block
            for item in batch:
                if 'data' in data[item['temp_id']]:
                    data[item['temp_id']]['data']['class'] = "Batch_Error"

        # Rate Limit Buffer & Periodic Checkpoint
        time.sleep(2.0) 
        if (batch_idx + 1) % 10 == 0:
            save_json(data, OUTPUT_FILE_PATH)

    # 4. Final Cleanup & Save
    for item in data:
        if 'temp_id' in item:
            del item['temp_id']
        
    save_json(data, OUTPUT_FILE_PATH)
    print(f"Pipeline complete! Results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()