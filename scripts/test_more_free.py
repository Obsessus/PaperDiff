import os, json, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

extra_free = [
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "minimax/minimax-m2.7:free",
    "poolside/laguna-s-2.1:free",
    "dots-studio/dots-3-note-preview:free"
]

prompt = """Compare these two papers on 'Methodology Rigor':
Paper A: 'We performed 10-fold cross validation with 5 ablation trials.'
Paper B: 'We tested on a single benchmark dataset without ablations.'

Respond ONLY with valid JSON:
{
  "paper_a": {"rating": 4.5, "explanation": "Rationale for A"},
  "paper_b": {"rating": 3.0, "explanation": "Rationale for B"},
  "signification": "A is better due to cross validation and ablation studies.",
  "better_paper": "Paper A"
}"""

for m in extra_free:
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=350
        )
        raw = resp.choices[0].message.content or ""
        clean = re.sub(r"^```[a-zA-Z]*\n?", "", raw.strip())
        clean = re.sub(r"\n?```$", "", clean).strip()
        data = json.loads(clean)
        print(f"[SUCCESS] {m} -> Adv: {data.get('better_paper')}")
    except Exception as e:
        print(f"[FAILED] {m} -> {str(e)[:90]}")
