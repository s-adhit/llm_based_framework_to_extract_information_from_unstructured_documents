# AI Text Classification Pipeline

This project is a robust, batch-processing pipeline designed to classify text data using Google's Gemini API. It handles rate limits, errors, and data persistence automatically.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
* Python 3.9 or higher.
* A Google Cloud Project with the **Gemini API** enabled.
* An API Key from [Google AI Studio](https://aistudio.google.com/).

### 2. Install Dependencies
It is recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Set up the API Key
Create a file named `.env` in the root directory of the project. Add your API key there. **Do not share this file.**

**File:** `.env`
```env
GOOGLE_API_KEY=your_actual_api_key_here
````

-----

## ⚙️ Configuration Guide (`src/config.py`)

You can customize the pipeline by editing `src/config.py`.

-----

### 1\. Changing the AI Model

To switch between Gemini models (e.g., **Flash** for speed, **Pro** for reasoning), update the `MODEL_NAME`.

```python
# src/config.py

# Faster, cheaper, good for high volume
MODEL_NAME = "gemini-2.5-flash" 

# Slower, more expensive, higher reasoning capability
# MODEL_NAME = "gemini-1.5-pro" 
```

### 2\. Adjusting Batch Size (Rate Limits)

The default batch size is tuned to optimize the Free Tier quota (1,500 requests/day or similar).

  * **Higher Batch Size (e.g., 200):** Uses fewer API requests. Best for Free Tier.
  * **Lower Batch Size (e.g., 50):** Easier to debug, but burns through quota request counts faster.

<!-- end list -->

```python
# src/config.py

# Recommended for Free Tier (process 10k items in ~50 requests)
BATCH_SIZE = 200 
```

### 3\. Prompting Strategies

You can change how the model is prompted by modifying `PROMPTING_STRATEGY`.

```python
# src/config.py

# Options: "zero_shot", "few_shot", "memory_prompt"
PROMPTING_STRATEGY = "memory_prompt"
```

  * `"zero_shot"`: The model is given **definitions only**. No examples are provided.
  * `"few_shot"`: The model is given a **static list of hardcoded examples** (`FEW_SHOT_EXAMPLES` in `config.py`) for guidance.
  * `"memory_prompt"`: The model is given definitions + the **last $N$ items** it successfully classified. This creates a feedback loop for style and consistency.

-----

## 🏃‍♂️ How to Run

1.  Place your input data in `data/input_data.json`.
      * *Format:* A list of objects, e.g., `[{"text": "Born in 1990..."}, ...]`.
2.  Run the main script:

<!-- end list -->

```bash
python main.py
```

3.  The script will generate `data/output_data.json`.
      * If the script stops, simply run `python main.py` again. It will detect the output file and resume automatically.

-----

## 📂 Project Structure

```text
.
├── .env                  # API Keys (Excluded from Git)
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── main.py               # Entry point of the script
├── data/
│   ├── input_data.json   # Source data
│   └── output_data.json  # Results (created automatically)
└── src/
    ├── config.py         # Global settings (Model, Batch Size, Categories)
    ├── gemini_client.py  # Handles API communication & Retry logic
    ├── prompt_builder.py # Constructs the prompt text
    └── utils.py          # Helper functions (File I/O, Memory Manager)
```

-----

## 📄 requirements.txt

Save the following content into a file named `requirements.txt`:

```text
google-genai
python-dotenv
tenacity
tqdm
```
