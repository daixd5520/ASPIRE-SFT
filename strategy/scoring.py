import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch
import argparse
import os
import json
import pickle

def load_model_and_tokenizer(model_name, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    return model, tokenizer

def load_data(file_path):
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    return pd.DataFrame(df)

def load_checkpoint(checkpoint_path):
    start_index = 0
    results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            start_index = checkpoint_data['index']
            results = checkpoint_data['results']
    return start_index, results

def save_checkpoint(checkpoint_path, index, results):
    checkpoint_data = {
        'index': index,
        'results': results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)

def save_results(output_path, results):
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

def process_batch(model, tokenizer, batch_df, device):
    prompts = batch_df['Instruction'].tolist()
    responses = batch_df['res'].tolist()
    resp_list = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}] for prompt, response in zip(prompts, responses)]
    resp_list = [tokenizer.apply_chat_template(resp, tokenize=False) for resp in resp_list]
    
    resp_batch = tokenizer(resp_list, return_tensors="pt", padding=True, truncation=True)
    resp_batch = {key: value.to(device) for key, value in resp_batch.items()}  

    with torch.no_grad():
        scores = model(resp_batch['input_ids'], attention_mask=resp_batch['attention_mask']).logits[:, 0].tolist()

    return scores

def main(args):
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

        batch_results = [] 
        for i, row in batch_df.iterrows():
            result = {
                'Index': row['Index'],
                'Instruction': row['Instruction'],
                'resScore': scores[i - batch_start]
            }
            results.append(result)
            batch_results.append(result)

        print(f"\nBatch {batch_start//batch_size + 1} result:")
        for res in batch_results:
            print(f"Index: {res['Index']}, score: {res['resScore']:.4f}")

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
