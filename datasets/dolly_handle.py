"""

transform dolly dataset into alpaca format.

"""
import json

input_file_path = "/data/d/SFT/datasets/sft/dolly.jsonl"
output_file_path = "/data/d/SFT/datasets/sft/dolly.json"

converted_data = []

with open(input_file_path, "r") as infile:
    for line in infile:
        entry = json.loads(line)
        
        new_entry = {
            "instruction":entry["instruction"],
            "input": entry['context'], 
            "output": entry["response"] 
        }
        
        converted_data.append(new_entry)

with open(output_file_path, "w") as outfile:
    json.dump(converted_data, outfile, indent=4)

print(f"转换完成，结果已保存到 {output_file_path}")