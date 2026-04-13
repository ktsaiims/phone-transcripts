import json
from pathlib import Path
from pprint import pprint

root_dir = Path(__file__).parent
data_dir = root_dir / 'raw_data'
combined_json = []

for file in data_dir.iterdir():
    if file.suffix == '.zip':
        continue
    if file.stat().st_size == 0:
        continue

    print(f'Loading file "{file.name}"')
    with file.open('r', encoding='utf-16') as f:
        messages = json.load(f)
    combined_json.append(messages)
print(f'Total combined JSON items: {len(combined_json)}')

output_path = data_dir / 'combined_json.json'
with output_path.open('w', encoding='utf-8') as f:
    json.dump(combined_json, f, indent=2)
print(f'Combined JSON: {output_path}')
