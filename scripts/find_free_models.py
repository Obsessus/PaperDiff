import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('OPENROUTER_API_KEY')

req = urllib.request.Request(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {key}'}
)
res = urllib.request.urlopen(req)
data = json.loads(res.read().decode())

all_models = data.get('data', [])
free_models = []

for m in all_models:
    pricing = m.get('pricing', {})
    prompt_p = float(pricing.get('prompt', 1))
    comp_p = float(pricing.get('completion', 1))
    if prompt_p == 0.0 and comp_p == 0.0:
        free_models.append(m)

print(f"Total Free Models on OpenRouter: {len(free_models)}")
print("=" * 80)
for m in free_models:
    mid = m['id']
    name = m.get('name', 'Unknown')
    ctx = m.get('context_length', 0)
    desc = m.get('description', '')[:90].replace('\n', ' ')
    print(f"ID: {mid}")
    print(f"  Name: {name} | Context: {ctx}")
    print(f"  Desc: {desc}...")
    print("-" * 60)
