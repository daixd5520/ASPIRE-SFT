
# from inference import InferencePipeline
# import torch

# models=['gemma2-2b','llama2-7b','llama2-13b']
# model_template_dict = {
#     'gemma2-2b': 'gemma',
#     'llama2-7b': 'llama2',
#     'llama2-13b': 'llama2'
# }
# datasets=['wizard']
# # model=models[0]
# model=models[2]
# dataset=datasets[0]
# merge_path=f"../lora_ckp/merged/{model}_{dataset}"
# model_paths=[merge_path+"1epo",merge_path+"2epo",merge_path]
# houzhui=["1epo","2epo",""]

models=['gemma2-2b','llama2-7b','llama2-13b']
model_template_dict = {
    'gemma2-2b': 'gemma',
    'llama2-7b': 'llama2',
    'llama2-13b': 'llama2'
}
datasets=['wizard','alpaca','dolly']
model=models[0]
dataset=datasets[1]
# merge_path=f"../lora_ckp/merged/{model}_{dataset}"
merge_path=f"../fft_ckp/{model}_{dataset}"
# model_paths=[merge_path+"1epo",merge_path+"2epo",merge_path]
model_paths=[merge_path+"/checkpoint-105",merge_path+"/checkpoint-211",merge_path]

houzhui=["1epo","2epo",""]

from inference import InferencePipeline,InferenceMultiCardPipeline
NUM=0
print(f"inferring {model_paths[NUM]}")
infer_output_file = f"../infer_res/fft/{model}{houzhui[NUM]}_{dataset}.pkl"
infer_checkpoint_dir = f"../infer_res/fft/checkpoints/{model}{houzhui[NUM]}_{dataset}.pkl"
infer_batch_size = 256
infer_num_gpus = 1
data_path=f"../datasets/sft/{dataset}.json"
infer_data_path = data_path

# pipeline = InferenceMultiCardPipeline(model_paths[NUM], infer_output_file, infer_checkpoint_dir, infer_batch_size)
pipeline = InferencePipeline(model_paths[NUM], infer_output_file, infer_checkpoint_dir, infer_batch_size, infer_num_gpus)
# pipeline = InferencePipeline('/home/pod/shared-nvme/d/SFT/models/llama2-7b', infer_output_file, infer_checkpoint_dir, infer_batch_size, infer_num_gpus)
pipeline.run_inference(data_path)
