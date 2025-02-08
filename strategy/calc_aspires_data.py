import pickle 
with open('../score_res/raw/llama2-7b_raw_wizard.pkl', 'rb') as f:
    score = pickle.load(f)
    
with open('../score_res_ground-truth/wizard-70k/scoring_res.pkl','rb') as f:
    gt_score = pickle.load(f)

import pandas as pd
import json
import os

datasets=['wizard','dolly']
dataset=datasets[0]
gt_score_df=pd.DataFrame(gt_score)
score_df=pd.DataFrame(score)

data = {}
for index, row in score_df.iterrows():
    idx = row['Index']
    score = row['resScore']
    if idx not in data:
        data[idx] = {}
    data[idx]["raw_score"] = score
    print(f"data[{idx}]['raw_score'] = {score}")

for index, row in gt_score_df.iterrows():
    idx = row['Index']
    score = row['resScore']
    data[idx]["gt_score"] = score
    print(f"data[{idx}]['gt_score'] = {score}")

score_diff = []
for idx, scores in data.items():
    score = scores.get("raw_score", -float('inf'))
    gt_score = scores.get("gt_score", -float('inf'))
    diff = gt_score - score
    score_diff.append([idx, diff])

score_diff_df = pd.DataFrame(score_diff, columns=["Index", "score_diff"])
# sort
score_diff_df = score_diff_df.sort_values(by="score_diff", ascending=True)
print(score_diff_df)

with open(f"../datasets/sft/{dataset}.json", "r") as f:
    original_data = json.load(f)

sorted_original_data = [original_data[idx] for idx in score_diff_df["Index"]]

# print(sorted_original_data)

output_file='../datasets/sft/wizard_llama2-13b_strategy2.json'
with open(output_file, "w") as f:
    json.dump(sorted_original_data, f, indent=4)
