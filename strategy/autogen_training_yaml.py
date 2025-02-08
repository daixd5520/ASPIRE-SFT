# autogen_training_yaml.py
"""
bash:
python autogen_training_yaml.py --yaml_prefix 'xxx/LLaMA-Factory/examples/train_lora/s1remove' --model_path 'xxx/Llama-2-7b-hf' --data_prefix 'wizard-70k' --remove_count '[4000,12000,20000,28000]' --save_path 'xxx/remove'

python:
from autogen_training_yaml import generate_yaml
generate_yaml(yaml_path, model_path, template, data_path, save_path)

"""
import os
import yaml
import argparse

def generate_yaml(yaml_path, model_path, template, data_path, save_path, epo=3.0):
    yaml_sections = {
        "model": {
            "model_name_or_path": model_path
        },
        "method": {
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": "all"
        },
        "dataset": {
            "dataset": f"{data_path}",
            "template": f"{template}",
            "cutoff_len": 1024,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16
        },
        "output": {
            "output_dir": f"{save_path}",
            "logging_steps": 10,
            "save_strategy": "epoch",
            "save_steps": 500
        },
        "train": {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1.0e-4,
            "num_train_epochs": epo, 
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "ddp_timeout": 180000000
        },
        "eval": {
            "val_size": 0.1,
            "per_device_eval_batch_size": 8,
            "eval_strategy": "steps",
            "eval_steps": 500
        }
    }

    comments = {
        "model": "### model",
        "method": "### method",
        "dataset": "### dataset",
        "output": "### output",
        "train": "### train",
        "eval": "### eval"
    }

    yaml_str = ""

    for section in ["model", "method", "dataset", "output", "train", "eval"]:
        yaml_str += f"{comments[section]}:\n"
        section_yaml = yaml.dump(yaml_sections[section], default_flow_style=False, sort_keys=False)
        
        indented_section = '\n'.join(['  ' + line if line.strip() else line for line in section_yaml.split('\n')])
        yaml_str += f"{indented_section}\n"

    file_name = f"{yaml_path}.yaml"
    with open(file_name, 'w') as file:
        file.write(yaml_str)

    print(f"Generated {file_name}")
    
def generate_yaml_fft(yaml_path, model_path, template, data_path, save_path, epo=3.0):
    yaml_sections = {
        "model": {
            "model_name_or_path": f"{model_path}",
            "trust_remote_code": True
        },
        "method": {
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full",
            "enable_liger_kernel": True,
            # "deepspeed": "xxx/deepspeed/ds_z3_config.json"
            "deepspeed": "/data/d/SFT/train_yamls/deepspeed/ds_z2_config.json"

        },
        "dataset": {
            "dataset": f"{data_path}",
            "template": f"{template}",
            "cutoff_len": 1024,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16
        },
        "output": {
            "output_dir": f"{save_path}",
            "logging_steps": 100,
            "save_strategy": "epoch",
            # "save_only_model": True,
            "save_steps": 500
        },
        "train": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-5,
            "num_train_epochs": epo,  
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": True,
            "ddp_timeout": 180000000
        },
        "eval": {
            "val_size": 0.1,
            "per_device_eval_batch_size": 1,
            "eval_strategy": "steps",
            "eval_steps": 500
        }
    }

    comments = {
        "model": "### model",
        "method": "### method",
        "dataset": "### dataset",
        "output": "### output",
        "train": "### train",
        "eval": "### eval"
    }

    yaml_str = ""

    for section in ["model", "method", "dataset", "output", "train", "eval"]:
        yaml_str += f"{comments[section]}:\n"
        section_yaml = yaml.dump(yaml_sections[section], default_flow_style=False, sort_keys=False)
        
        indented_section = '\n'.join(['  ' + line if line.strip() else line for line in section_yaml.split('\n')])
        yaml_str += f"{indented_section}\n"

    file_name = f"{yaml_path}.yaml"
    with open(file_name, 'w') as file:
        file.write(yaml_str)

    print(f"Generated {file_name}")

def main():
    parser = argparse.ArgumentParser(description="Generate YAML files based on command line arguments.")
    parser.add_argument("--yaml_prefix", type=str, help="Path prefix of your yaml.")
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--data_prefix", type=str, help="Prefix for the dataset.")
    parser.add_argument("--remove_count", type=str, help="List of remove counts, e.g., '[4,5,6,7,8]'.")
    parser.add_argument("--save_path", type=str, help="Prefix for the save path.")

    args = parser.parse_args()

    remove_count = list(map(int, args.remove_count.strip('[]').split(',')))

    generate_yaml(args.yaml_prefix, args.model_path, args.data_prefix, remove_count, args.save_path)

if __name__ == "__main__":
    main()
