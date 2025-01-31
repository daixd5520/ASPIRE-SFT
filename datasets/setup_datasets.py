"""
download alpaca,wizard,dolly datasets in one file.
"""
import subprocess
import os

def run_command(command, description):
    print(f"Running: {description}")
    process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Output:\n{process.stdout.decode()}")
    if process.stderr:
        print(f"Errors:\n{process.stderr.decode()}")
    print(f"Finished: {description}\n")

def main():
    download_path = "./sft/"
    
    # 确保下载路径存在
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    commands = [
        (f"wget -P {download_path} https://hf-mirror.com/datasets/WizardLMTeam/WizardLM_evol_instruct_70k/resolve/main/alpaca_evol_instruct_70k.json?download=true", "Downloading wizard.json"),
        (f"wget -P {download_path} https://ghp.ci/github.com/tatsu-lab/stanford_alpaca/raw/refs/heads/main/alpaca_data.json", "Downloading alpaca.json"),
        (f"wget -P {download_path} https://hf-mirror.com/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl?download=true", "Downloading dolly.jsonl"),
        (f"mv {download_path}alpaca_evol_instruct_70k.json?download=true {download_path}wizard.json", "Renaming wizard.json"),
        (f"mv {download_path}alpaca_data.json {download_path}alpaca.json", "Renaming alpaca.json"),
        (f"mv {download_path}databricks-dolly-15k.jsonl?download=true {download_path}dolly.jsonl", "Renaming dolly.jsonl")
    ]

    for command, description in commands:
        print(command)
        run_command(command, description)

if __name__ == "__main__":
    main()