"""
transform wizard data into alpaca format.
"""
import json

with open('./sft/wizard.json', 'r') as file:
    data = json.load(file)

for item in data:
    item['input'] = ""

with open('./sft/wizard.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Data updated and saved into ./sft/wizard.json")
