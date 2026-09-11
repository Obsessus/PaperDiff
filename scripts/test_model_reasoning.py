import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

candidate_models = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "z-ai/glm-5.2:free",
    "minimax/minimax-m3:free",
    "openrouter/free",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat"
]

test_prompt = """Compare these two papers on 'Methodology Rigor':
Paper A: 'We performed 10-fold cross validation with 5 ablation trials.'
Paper B: 'We tested on a single benchmark dataset without ablations.'

Respond ONLY with valid JSON:
{
  "paper_a": {"rating": 4.5, "explanation": "Rationale for A"},
  "paper_b": {"rating": 3.0, "explanation": "Rationale for B"},
  "signification": "A is better due to cross validation and ablation studies.",
  "better_paper": "Paper A"
}"""

print("Benchmarking Candidate Models for Document Comparison & Reasoning:\n")

for m in candidate_models:
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.1,
            max_tokens=350
        )
        raw = resp.choices[0].message.content or ""
        # Clean markdown codeblocks
        clean = raw.strip()
        clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
        clean = clean.strip()
        
        data = json.loads(clean)
        print(f"[SUCCESS] Model: {m}")
        print(f"   -> Paper A: {data.get('paper_a', {}).get('rating')} | Paper B: {data.get('paper_b', {}).get('rating')}")
        print(f"   -> Advantage: {data.get('better_paper')}")
        print(f"   -> Signification: {data.get('signification')[:90]}...")
        print("-" * 60)
    except Exception as e:
        print(f"[FAILED] Model: {m} -> Error: {str(e)[:120]}")
        print("-" * 60)
