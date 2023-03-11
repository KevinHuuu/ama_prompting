# Ask Me Anything: A simple strategy for prompting language models

![GitHub](https://img.shields.io/github/license/HazyResearch/ama_prompting)
[![Together AI](https://together.xyz/assets/images/ai_platform.svg)](https://together.xyz/)

This repository contains code for the Ask Me Anything (AMA) prompt-aggregation strategy. The end-to-end AMA approach includes (1) recursively using the language model to transform the task format and prompt and (2) aggregating the predictions of multiple prompts using weak supervision. We include code for both components and pointers to the publicly downloadable datasets. See our [paper](https://arxiv.org/abs/2210.02441) for more details.

<p align="center"><img width="95%" src="imgs/decomp.png" /></p>



## Table of Contents
- [Setup](#setup)
- [Data](#getting-the-data)
- [Running models](#models)
- [Running experiments](#experiments)
- [Repository Structure](#overall-repository-structure)
- [Citation](#citation)


## Setup

### Installation
Here we will setup the AMA code (prompting models for tasks), weak supervision code (aggregating predictions), and [Manifest](https://github.com/HazyResearch/manifest/) code (tooling for easily loading and running the models).

We encourage the use of conda environments:
```
conda create --name ama python=3.8
conda activate ama
```

Clone as follows:
```bash
# Ask Me Anything code
git clone git@github.com:HazyResearch/ama_prompting.git
cd ama_prompting
pip install -r requirements.txt

# Manifest 
git clone git@github.com:HazyResearch/manifest.git
cd manifest
pip install -e .
```


### Getting the data
We assume all data lives in the ```AMA_DATA``` environment variable. By default, this is set to ```/home/data```. To change this, run
```bash
export AMA_DATA=<path>
```

Please follow the instructions below to download all necessary data for experiments. 
 
1. Download the PromptSource (P3) dataset from HuggingFace at https://huggingface.co/datasets/bigscience/P3.
```bash
cd $AMA_DATA
git lfs install
git clone https://huggingface.co/datasets/bigscience/P3
```
Then run [ama_prompting/download_p3.py](./download_p3.py). We use the GPT3-Style prompts in the few-shot baseline for each benchmark.

2. We downloaded the remaining tasks from the following sources:
    * [AGNews, DBPedia, and SST2](https://github.com/tonyzhaozh/few-shot-learning)
    * [Amazon Products](https://github.com/allenai/flex/blob/75d6d1cea66df2c8a7e3d429c6af5008ccf1544b/fewshot/hf_datasets_scripts/amazon/amazon.py)
    * [Natural Questions and WebQs](https://github.com/facebookresearch/FiD)
    * [RealTimeQA](https://github.com/realtimeqa/realtimeqa_public/tree/main/past/2022) (GCS files from June 17th - July 22, 2022)
    * [ReCoRD](https://sheng-z.github.io/ReCoRD-explorer/)
    * [StoryCloze](http://goo.gl/forms/aQz39sdDrO)

3. Experiments

### CB full run
```
export DAAS_URL=<your_daas_url>
export DAAS_KEY=<your_daas_key>
python3 tasks/CB_final.py \
    --num_boost 1 \
    --output_metrics_file ../ama_logs/metrics.jsonl \
    --cache_connection_question ../ama_logs/question_manifest_cache.sqlite \
    --cache_connection_answer ../ama_logs/answer_manifest_cache.sqlite \
    --save_dir ../ama_logs/ama_final_runs --boost_train_examples 0
```


The accuracy is 82.1%
```
Saving to ../ama_logs/ama_final_runs/super_glue_cb/QUESTION_gpt2_ANSWER_gpt2_decomposed_03082023.json
Accuracy by Boost Set Decomposed [0.8214285714285714]
Accuracy by Boost Set Decomposed Average 0.8214285714285714
Accuracy Boost Decomposed 0.8214285714285714
Saved metrics to ../ama_logs/metrics.jsonl
Saved final data to ../ama_logs/ama_final_runs/super_glue_cb
```

### AGNews example run

```
python3 tasks/AGNews_final.py \
    --run_zeroshot 1\
    --num_boost 3 \
    --output_metrics_file ../ama_logs/metrics.jsonl \
    --cache_connection_question ../ama_logs/question_manifest_cache.sqlite \
    --cache_connection_answer ../ama_logs/answer_manifest_cache.sqlite \
    --save_dir ../ama_logs/ama_final_runs  --num_run 1 --boost_train_examples 0
```
    
