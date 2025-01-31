import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch
import argparse
import os
import json
import pickle

def load_model_and_tokenizer(model_name, device):
    """加载模型和tokenizer"""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 使用DataParallel支持多卡
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    return model, tokenizer

def load_data(file_path):
    """加载数据"""
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    return pd.DataFrame(df)

def load_checkpoint(checkpoint_path):
    """加载检查点"""
    start_index = 0
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            start_index = checkpoint_data['index']
            results = checkpoint_data['results']
    return start_index, results

def save_checkpoint(checkpoint_path, index, results):
    """保存检查点"""
    checkpoint_data = {
        'index': index,
        'results': results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)

def save_results(output_path, results):
    """保存结果"""
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

def process_batch(model, tokenizer, batch_df, device):
    """处理批量数据"""
    prompts = batch_df['Instruction'].tolist()
    responses = batch_df['res'].tolist()
    resp_list = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}] for prompt, response in zip(prompts, responses)]
    resp_list = [tokenizer.apply_chat_template(resp, tokenize=False) for resp in resp_list]
    
    # 将输入张量移动到指定设备
    resp_batch = tokenizer(resp_list, return_tensors="pt", padding=True, truncation=True)
    resp_batch = {key: value.to(device) for key, value in resp_batch.items()}  # 确保所有张量都在同一设备上

    with torch.no_grad():
        # 使用DataParallel时，直接调用model
        scores = model(resp_batch['input_ids'], attention_mask=resp_batch['attention_mask']).logits[:, 0].tolist()

    return scores

def main(args):
    """主函数"""
    file_path = args.file
    ckp_dir = args.ckp_dir
    output_path = args.out_dir
    checkpoint_path = os.path.join(ckp_dir, 'checkpoint.json')

    os.makedirs(ckp_dir, exist_ok=True)
    device = torch.device("cuda:0")

    model_name = "../models/URM-LLaMa-3.1-8B"
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    df = load_data(file_path)
    start_index, results = load_checkpoint(checkpoint_path)

    batch_size = 512

    for batch_start in tqdm(range(start_index, df.shape[0], batch_size), desc="Processing Batches"):
        batch_end = min(batch_start + batch_size, df.shape[0])
        batch_df = df.iloc[batch_start:batch_end]

        scores = process_batch(model, tokenizer, batch_df, device)

        batch_results = []  # 存储当前batch的结果
        for i, row in batch_df.iterrows():
            result = {
                'Index': row['Index'],
                'Instruction': row['Instruction'],
                'res得分': scores[i - batch_start]
            }
            results.append(result)
            batch_results.append(result)

        # 打印当前batch的结果
        print(f"\nBatch {batch_start//batch_size + 1} 结果:")
        for res in batch_results:
            print(f"Index: {res['Index']}, 得分: {res['res得分']:.4f}")

        save_checkpoint(checkpoint_path, batch_end, results)

    save_results(output_path, results)
    print(f"Scoring completed and results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process model responses.")
    parser.add_argument('--file', required=True, help="File path(path to .pkl)")
    parser.add_argument('--ckp_dir', required=True, help="Checkpoint path(path to folder)")
    parser.add_argument('--out_dir', required=True, help="Output path(path to folder )")
    args = parser.parse_args()
    main(args)