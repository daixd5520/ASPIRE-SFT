# 1. Data Preparation
```
cd datasets
python setup_datasets.py
python wizard_handle.py # we use alpaca data format so you should run transform script first.
python dolly_handle.py
```
# 2. ASPIRE-D
## 2.1 train
We use LLaMA_Factory to perform finetuning process. Go to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) for details.
```
llamafactory-cli train train_yamls/example.yaml 
```
## 2.2 infer
Infer sft datasets using each epoch checkpoints.
```
python inf1.py
```
## 2.3 RM scoring
Score each inference result generated in step 3.
```
python score1.py
```
## 2.4. get ASPIRE-D data
```
python calc_aspired_data.py
```

# 3 ASPIRE-S
## 3.1 infer
Infer sft datasets using pretrained LLM.( `pipeline = InferencePipeline(model_path, infer_output_file, infer_checkpoint_dir, infer_batch_size, infer_num_gpus)`)
```
python inf1.py
```
## 3.2 RM scoring
1) Score inference result generated in step 3.
2) Score original SFT dataset (Ground Truth).
```
python score1.py
```
## 3.3 get ASPIRE-S data
```
python calc_aspires_data.py
```