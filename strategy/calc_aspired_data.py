models=['gemma2-2b','llama2-7b','llama2-13b']
model_template_dict = {
    'gemma2-2b': 'gemma',
    'llama2-7b': 'llama2',
    'llama2-13b': 'llama2'
}
datasets=['wizard','dolly','alpaca']
model=models[1]#TO Modify
dataset=datasets[2]
merge_path=f"../lora_ckp/merged/{model}_{dataset}"
model_paths=[merge_path+"1epo",merge_path+"2epo",merge_path]

data_path=f"../datasets/sft/{dataset}.json"
houzhui=["1epo","2epo",""]

import pandas as pd
import json
import os
score_files = [f"../score_res/lora/URM_llama/{model}{h}_{dataset}.pkl" for h in houzhui]

data = {}
for i, file in enumerate(score_files):
    if os.path.exists(file):
        df = pd.read_pickle(file)
        if isinstance(df, list):
            df = pd.DataFrame(df)
        for index, row in df.iterrows():
            idx = row['Index']
            score = row['res得分']
            if idx not in data:
                data[idx] = {}
            data[idx][houzhui[i]] = score

result = []
for idx, scores in data.items():
    max_score = max(scores.values())
    if max_score == scores.get("1epo", -float('inf')):
        result.append([idx, "1"])
    elif max_score == scores.get("2epo", -float('inf')):
        result.append([idx, "2"])
    elif max_score == scores.get("", -float('inf')):
        result.append([idx, "3"])

result_df = pd.DataFrame(result, columns=["Index", "epo"])

output_file = f"data_index_epo_{model}_{dataset}.pkl"
result_df.to_pickle(output_file)

print(f"DataFrame saved to {output_file}")
# print(result_df)

with open(f"../datasets/sft/{dataset}.json", "r") as f:
    original_data = json.load(f)

data_1epo = original_data
data_2epo = []
data_3epo = []

for index, row in result_df.iterrows():
    idx = row['Index']
    epo = row['epo']
    if epo == "2":
        data_2epo.append(original_data[int(idx)])
    elif epo == "3":
        data_3epo.append(original_data[int(idx)])

data_2epo.extend(data_3epo)

with open(f"../datasets/sft/{dataset}_{model}1epo.json", "w") as f:
    json.dump(data_1epo, f, indent=4)

with open(f"../datasets/sft/{dataset}_{model}2epo.json", "w") as f:
    json.dump(data_2epo, f, indent=4)

with open(f"../datasets/sft/{dataset}_{model}3epo.json", "w") as f:
    json.dump(data_3epo, f, indent=4)

# 输出三个文件的数据行数
# print(f"../datasets/sft/{dataset}_{model}1epo.json 行数: {len(data_1epo)}")
# print(f"../datasets/sft/{dataset}_{model}2epo.json 行数: {len(data_2epo)}")
# print(f"../datasets/sft/{dataset}_{model}3epo.json 行数: {len(data_3epo)}")

print("ASPIRE-D数据生成完成!")