import os
import pickle
from tqdm import tqdm
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
# from flash_attn import FlashAttention

class InferencePipeline:
    def __init__(self, model_path, output_file, checkpoint_dir, batch_size, num_gpus):
        self.model_path = model_path
        self.output_file = output_file
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.devices = [f'cuda:{i}' for i in range(num_gpus)]
        
        # 首先创建tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'  # 设置为左侧填充
        
        # 创建模型并立即移动到对应的GPU
        self.models = []
        for i in range(num_gpus):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=self.devices[i]  # 直接指定设备
            ).to('cuda')
            model.eval()
            self.models.append(model)

    
    # def gen_batch(self, contents, num_responses=1):
    #     input_texts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{content}\n\n### Response:\n" for content in contents]
    #     inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
    #     # Split inputs across multiple GPUs
    #     split_inputs = [{} for _ in range(self.num_gpus)]
    #     for key in inputs:
    #         split_inputs[0][key] = inputs[key][:, :inputs[key].shape[1] // self.num_gpus].to(self.devices[0])
    #         for i in range(1, self.num_gpus):
    #             split_inputs[i][key] = inputs[key][:, (inputs[key].shape[1] // self.num_gpus) * i:(inputs[key].shape[1] // self.num_gpus) * (i + 1)].to(self.devices[i])
    def gen_batch(self, contents, num_responses=1):
        input_texts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{content}\n\n### Response:\n" for content in contents]
        inputs = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # 确保输入数据在正确的设备上
        if self.num_gpus == 1:
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            split_inputs = [inputs]
        else:
            # 多GPU情况保持原有逻辑
            split_inputs = [{} for _ in range(self.num_gpus)]
            for key in inputs:
                split_inputs[0][key] = inputs[key][:, :inputs[key].shape[1] // self.num_gpus]
                for i in range(1, self.num_gpus):
                    split_inputs[i][key] = inputs[key][:, (inputs[key].shape[1] // self.num_gpus) * i:(inputs[key].shape[1] // self.num_gpus) * (i + 1)].to(self.devices[i])
        
    # ... rest of the code ...
        generated_answers = [[] for _ in range(len(contents))]
        for _ in range(num_responses):
            outputs = []
            for i in range(self.num_gpus):
                with torch.no_grad():
                    outputs.append(self.models[i].generate(**split_inputs[i], max_new_tokens=128, do_sample=False))
            
            for i in range(len(contents)):
                if self.num_gpus == 1:
                    # 单GPU情况直接使用输出
                    generated_answer = self.tokenizer.decode(outputs[0][i].to('cuda'), skip_special_tokens=True)
                else:
                    # 多GPU情况保持原有逻辑
                    outputs_on_same_device = [outputs[j][i].to(self.devices[0]) for j in range(self.num_gpus)]
                    generated_answer = self.tokenizer.decode(torch.cat(outputs_on_same_device, dim=0), skip_special_tokens=True)
                # 截断生成的响应，移除指令部分
                response_start = generated_answer.find("### Response:\n") + len("### Response:\n")
                generated_answer = generated_answer[response_start:].strip()
                generated_answers[i].append(generated_answer)
        
        return generated_answers
    

    def run_inference(self, data_path):
        # Load Alpaca data
        with open(data_path, 'r') as file:
            alpaca_data = json.load(file)

        # Prepare the checkpoint directory
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pkl')

        # 准备检查点目录
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pkl')

        # 检查是否存在检查点
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                results = pickle.load(f)
            print(f"已加载 {len(results)} 条结果从检查点。")
        else:
            results = []
            print("未找到检查点，从头开始。")

        start_index = len(results)
        print(f"从索引 {start_index} 开始处理，总数据量 {len(alpaca_data)}")

        # Process data with progress bar and save after each inference
        for idx in tqdm(range(start_index, len(alpaca_data), self.batch_size), desc="Processing Data"):
            # batch_instructions = [entry['instruction'] for entry in alpaca_data[idx:idx + self.batch_size]]
            batch_end = min(idx + self.batch_size, len(alpaca_data))
            batch_instructions = [entry['instruction'] for entry in alpaca_data[idx:batch_end]]
                
            # Generate responses on the GPUs
            response_1epo_batch = self.gen_batch(batch_instructions)
            
            # Append the result to the results list
            for i, instruction in enumerate(batch_instructions):
                results.append({
                    'Index': idx + i,
                    'Instruction': instruction,
                    'res': response_1epo_batch[i][0]
                })
            
            # Save the results to the pickle file
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(results, f)
            """
            format:
                        'Index': idx + i,
                        'Instruction': instruction,
                        'res': response_1epo_batch[i][0]
            
            """
            print(f"已保存检查点，当前处理了 {len(results)} 条数据")

        # Convert the results to a DataFrame and save to Excel
        result_df = pd.DataFrame(results)
        # 直接存储 DataFrame 到 pickle 文件
        result_df.to_pickle(self.output_file)

        print(f"结果已保存到 {self.output_file}")

class InferenceMultiCardPipeline:
    def __init__(self, model_path, output_file, checkpoint_dir, batch_size):
        self.model_path = model_path
        self.output_file = output_file
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 使用device_map="auto"自动处理模型切分
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",  # 自动决定如何跨设备切分模型
            torch_dtype=torch.bfloat16,
        )

    def gen_batch(self, contents, num_responses=1):
        input_texts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{content}\n\n### Response:\n" for content in contents]
        inputs = self.tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        generated_answers = [[] for _ in range(len(contents))]
        for _ in range(num_responses):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            for i in range(len(contents)):
                generated_answer = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                response_start = generated_answer.find("### Response:\n") + len("### Response:\n")
                generated_answer = generated_answer[response_start:].strip()
                generated_answers[i].append(generated_answer)
        
        return generated_answers

    def run_inference(self, data_path):
        # Load Alpaca data
        with open(data_path, 'r') as file:
            alpaca_data = json.load(file)

        # 准备检查点目录
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.pkl')

        # 检查是否存在检查点
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                results = pickle.load(f)
            print(f"已加载 {len(results)} 条结果从检查点。")
        else:
            results = []
            print("未找到检查点，从头开始。")

        start_index = len(results)
        print(f"从索引 {start_index} 开始处理，总数据量 {len(alpaca_data)}")

        try:
            for idx in tqdm(range(start_index, len(alpaca_data), self.batch_size), desc="处理数据中"):
                batch_end = min(idx + self.batch_size, len(alpaca_data))
                batch_instructions = [entry['instruction'] for entry in alpaca_data[idx:batch_end]]
                
                # 生成回答
                response_batch = self.gen_batch(batch_instructions)
                
                # 保存结果
                for i, instruction in enumerate(batch_instructions):
                    results.append({
                        'Index': idx + i,
                        'Instruction': instruction,
                        'res': response_batch[i][0]
                    })
                
                # 保存检查点
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(results, f)
                
                print(f"已保存检查点，当前处理了 {len(results)} 条数据")

        except Exception as e:
            print(f"发生错误: {str(e)}")
            print("保存当前进度...")
        
        finally:
            # 保存最终结果
            result_df = pd.DataFrame(results)
            result_df.to_pickle(self.output_file)
            print(f"结果已保存到 {self.output_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./gemma-2-2b",
        help="model path.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./gemma2-wizard-full_bo1.pkl",
        help="Output file path.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./infer_res/gemma2/wizard/1006/checkpoint/full",
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs to use.",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    pipeline = InferencePipeline(args.model, args.output_file, args.checkpoint_dir, args.batch_size, args.num_gpus)
    pipeline.run_inference('./alpaca_evol_instruct_70k.json')

if __name__ == "__main__":
    main()