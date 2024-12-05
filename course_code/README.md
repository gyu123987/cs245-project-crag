# CS245 Crag Project: Part 1

## Models

We did many experiments on many different methods and types of models, so there are many model and output files. We decided to keep the code files of each of the models in the repository because they are mentioned in the report.

However our final model is the `rag_baseline_final_model.py` which we have instructions on running with the `meta-llama/Llama-3.2-1B-Instruct` model.

## Running our Project

### Download Dataset
1. Download the dataset from [here](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/problems/retrieval-summarization/dataset_files) and place `crag_task_1_dev_v4_release.jsonl.bz2` into the `data` directory

### Setting up Environment
1. Run `conda create -n crag python=3.10` to create a conda environment
2. Active the environment with `conda activate crag`
3. Install all requirements 
```
pip install -r requirements.txt
pip install --upgrade openai
```
4. Set up the following environment variables
```
huggingface-cli login --token "your_access_token"
export CUDA_VISIBLE_DEVICES=0
```

### Running Generate
1. Start up VLLM 
```
vllm serve meta-llama/Llama-3.2-1B-Instruct --gpu_memory_utilization=0.85 --tensor_parallel_size=1 --dtype="half" --port=8088 --enforce_eager
```
2. Open a new terminal, activate your conda environment, and generate the predictions
```
python generate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --split 1 --model_name "rag_baseline_final_model" --llm_name "meta-llama/Llama-3.2-1B-Instruct" --is_server --vllm_server "http://localhost:8088/v1" 
```

### Running Evaluate
1. Start up VLLM if not started in previous step
```
vllm serve meta-llama/Llama-3.2-1B-Instruct --gpu_memory_utilization=0.85 --tensor_parallel_size=1 --dtype="half" --port=8088 --enforce_eager
```
3. Open a new terminal, activate your conda environment, and generate the predictions (or can reuse terminal from generating)
```
python evaluate.py --dataset_path "data/crag_task_1_dev_v4_release.jsonl.bz2" --model_name "rag_baseline_final_model” --llm_name "meta-llama/Llama-3.2-1B-Instruct" --is_server --vllm_server "http://localhost:8088/v1" --max_retries 10
```
