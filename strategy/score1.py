
models=['gemma2-2b','llama2-7b','llama2-13b']
model_template_dict = {
    'gemma2-2b': 'gemma',
    'llama2-7b': 'llama2',
    'llama2-13b': 'llama2'
}
datasets=['wizard','dolly']
# TODO @JJF 跑gemma2-2b_wizard、llama2-7b_wizard、llama2-13b_wizard一共三个组合;
model=models[0]# TODO @JJF 选择
dataset=datasets[0]# TODO @JJF 选择,后面代码不用动

houzhui=["1epo","2epo",""]

NUM=0
import os
from scoring import main
# from rmbert_scoring import rmbert_main

# 设置可见的 GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这里设置为 "0"，表示只使用 GPU 0

print(f"scoring {model}{houzhui[NUM]}_{dataset}")
infer_output_file = f"../infer_res/lora/{model}{houzhui[NUM]}_{dataset}.pkl"
scoring_output_file = f"../score_res/lora/{model}{houzhui[NUM]}_{dataset}.pkl"
scoring_checkpoint_dir = f"../score_res/lora/checkpoints/{model}{houzhui[NUM]}_{dataset}.pkl"


# /data/d/SFT/infer_res/raw/llama2-13b_raw_wizard.pkl
# print(f"scoring {model}_raw_{dataset}")
# infer_output_file = f"../infer_res/raw/{model}_raw_{dataset}.pkl"
# scoring_output_file = f"../score_res/{model}_raw_{dataset}.pkl"
# scoring_checkpoint_dir = f"../score_res/checkpoints/{model}_raw_{dataset}.pkl"

# # 新RM
# infer_output_file = f"../infer_res/lora/{model}{houzhui[NUM]}_{dataset}.pkl"
# scoring_output_file = f"../score_res/lora/rmbert/{model}{houzhui[NUM]}_{dataset}.pkl"
# scoring_checkpoint_dir = f"../score_res/lora/rmbert/checkpoints/{model}{houzhui[NUM]}_{dataset}.pkl"
# # TODO @JJF 查看scoring_output_file有没有正常出结果



from argparse import Namespace
args = {
    'file': infer_output_file,
    'ckp_dir': scoring_checkpoint_dir,
    'out_dir': scoring_output_file
}
# 将字典转换为 argparse.Namespace 对象
args = Namespace(**args)
# main(args)
rmbert_main(args)