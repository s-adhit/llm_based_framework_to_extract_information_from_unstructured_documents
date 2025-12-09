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
    """
    Ensures the output file exists.
    If it doesn't, it copies the input data to the output path.
    """
    if os.path.exists(OUTPUT_FILE_PATH):
        print(f"Resuming from existing output file: {OUTPUT_FILE_PATH}")
        return load_json(OUTPUT_FILE_PATH)
    else:
        print(f"No output file found. Creating {OUTPUT_FILE_PATH} from input...")
        data = load_json(INPUT_FILE_PATH)
        if not data:
            raise ValueError(f"Input file {INPUT_FILE_PATH} is empty or missing!")
        
        # Save the initial copy to the output path
        save_json(data, OUTPUT_FILE_PATH)
        return data

def main():
    # 1. Load Data
    data = initialize_output_file()
    
    # 2. Identify Pending Items & Initialize Memory
    pending_items = []
    for i, item in enumerate(data):
        if "class" not in item:
            item['temp_id'] = i
            pending_items.append(item)
    
    print(f"Strategy: {PROMPTING_STRATEGY.upper()}")
    
    # Initialize Memory ONLY if needed
    if PROMPTING_STRATEGY == "memory_prompt":
        memory = MemoryManager()
    else:
        memory = None

    if not pending_items:
        print("All items are already classified. Done!")
        return

    # 3. Batch Processing
    chunks = list(chunk_data(pending_items, BATCH_SIZE))
    
    for batch_idx, batch in tqdm(enumerate(chunks), total=len(chunks), desc="Processing Batches"):
        
        try:
            # --- CONTEXT RETRIEVAL ---
            dynamic_context = None
            if PROMPTING_STRATEGY == "memory_prompt" and memory is not None:
                dynamic_context = memory.get_examples()
                
            prompt = build_batch_classification_prompt(batch, dynamic_memory=dynamic_context)
            
            # --- API CALL ---
            results = get_batch_classification(batch, prompt)
            
            # --- RESULTS PROCESSING ---
            result_map = {r.get('id'): r for r in results if 'id' in r}
            
            # Update the main data list
            for item in batch:
                t_id = item['temp_id']
                
                if t_id in result_map:
                    res = result_map[t_id]
                    # MAP ABBREVIATIONS BACK TO FULL NAMES
                    item['class'] = res.get('c', "Uncertain") 
                    item['confidence_score'] = res.get('s', 0.0)
                    
                    # Update the in-memory data object as well
                    data[t_id]['class'] = item['class']
                    data[t_id]['confidence_score'] = item['confidence_score']
                else:
                    data[t_id]['class'] = "API_Missed"
                    data[t_id]['confidence_score'] = 0.0

            # --- MEMORY UPDATE (Feedback Loop) ---
            if PROMPTING_STRATEGY == "memory_prompt" and memory is not None:
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "id": r.get("id"),
                        "class": r.get("c"), # Map 'c' to 'class' for memory
                        "confidence_score": r.get("s")
                    })
                memory.add_batch(batch_inputs=batch, batch_results=formatted_results)

        except Exception as e:
            print(f"\nBatch {batch_idx} failed: {e}")
            for item in batch:
                data[item['temp_id']]['class'] = "Batch_Error"

        # Rate Limit Sleep & Checkpoint (Your existing logic)
        time.sleep(7.0)
        if (batch_idx + 1) % 5 == 0:
            save_json(data, OUTPUT_FILE_PATH)

    # 4. Final Cleanup & Save
    for item in data:
        if 'temp_id' in item:
            del item['temp_id']
        
    save_json(data, OUTPUT_FILE_PATH)
    print(f"Pipeline complete! Results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()