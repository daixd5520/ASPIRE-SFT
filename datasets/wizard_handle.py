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

print("数据已更新并保存到 ./sft/wizard.json")