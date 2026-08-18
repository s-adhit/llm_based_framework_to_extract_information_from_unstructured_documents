#main.py
import os
import time
from tqdm import tqdm
from src.config import BATCH_SIZE, PROMPTING_STRATEGY, INPUT_FILE_PATH, OUTPUT_FILE_PATH
from src.utils import load_json, save_json
from src.prompt_builder import build_batch_classification_prompt
from src.gemini_client import get_batch_classification

def chunk_data(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]

def initialize_output_file():
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
        if not category or category == ["API_Missed"] or category == ["Batch_Error"]:
            item['temp_id'] = i
            pending_items.append(item)

    print(f"Strategy: {PROMPTING_STRATEGY.upper()} | Pending: {len(pending_items)}")

    if not pending_items:
        print("All items are already classified. Done!")
        return

    # 3. Batch Processing
    chunks = list(chunk_data(pending_items, BATCH_SIZE))
    previous_sentence = None

    for batch_idx, batch in tqdm(enumerate(chunks), total=len(chunks), desc="Processing"):
        try:
            # Reset context at document boundaries
            if batch[0].get('data', {}).get('sent_id') == 0:
                previous_sentence = None

            # --- PROMPT & API CALL ---
            prompt = build_batch_classification_prompt(batch, previous_sentence=previous_sentence)
            results = get_batch_classification(batch, prompt)

            result_map = {str(r.get('id')): r for r in results if r.get('id') is not None}

            # --- RESULTS PROCESSING ---
            missed_in_this_batch = 0
            for item in batch:
                t_idx = item['temp_id']
                inner_data = item.get('data', {})
                raw_id = inner_data.get('sent_id')
                sent_id_str = str(raw_id)

                if sent_id_str in result_map:
                    res = result_map[sent_id_str]
                    item['data']['category'] = res.get('c', ["Others"])
                else:
                    item['data']['category'] = ["API_Missed"]
                    missed_in_this_batch += 1

            if missed_in_this_batch > 0:
                print(f"\n[!] Batch {batch_idx}: {missed_in_this_batch} missed. Looked for '{sent_id_str}'")
            else:
                print(f"\n[✓] Batch {batch_idx}: 100% Match.")

            # Update previous sentence for next batch
            previous_sentence = batch[-1].get('data', {}).get('text')

            # --- IMMEDIATE SAVE ---
            save_json(data, OUTPUT_FILE_PATH)

        except Exception as e:
            print(f"\nBatch {batch_idx} failed: {e}")
            for item in batch:
                if 'data' in data[item['temp_id']]:
                    data[item['temp_id']]['data']['category'] = ["Batch_Error"]
            save_json(data, OUTPUT_FILE_PATH)

        time.sleep(0.5)

    print(f"Pipeline complete! Results saved to {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()