from pathlib import Path
from collections import Counter
import json
from datasets import load_dataset
import re
import pandas as pd
from typing import Callable, List
from manifest import Manifest
from flask import Flask
import json
import requests
import ssl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
import os
import sqlite3

def create_cache_table():
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS cache (prompt TEXT PRIMARY KEY, completion TEXT)')
    conn.commit()
    conn.close()

def cache_response(prompt, completion):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO cache (prompt, completion) VALUES (?, ?)', (prompt, completion))
    conn.commit()
    conn.close()

def get_cached_response(prompt):
    conn = sqlite3.connect('cache.db')
    c = conn.cursor()
    c.execute('SELECT completion FROM cache WHERE prompt=?', (prompt,))
    result = c.fetchone()
    conn.close()
    if result is not None:
        return result[0]
    else:
        return None

class InputOutputPrompt:
    def __init__(self,
        input_formatter: Callable,
        output_formatter: Callable,
        required_keys: List,
        input_output_sep: str = "\n",
        example_sep: str = "\n\n",
        instruction: str = ""
    ):
        self.input_formatter = input_formatter
        self.output_formatter = output_formatter
        self.required_keys = required_keys
        self.input_output_sep = input_output_sep
        self.example_sep = example_sep
        self.instruction = instruction

    def __call__(self, input_output_pairs: pd.DataFrame):
        examples = []
        for _, example in input_output_pairs.iterrows():
            examples.append(f"{self.input_formatter(example)}{self.input_output_sep}{self.output_formatter(example)}")
        if examples:
            input_str = self.example_sep.join(examples)
            res = f"{self.instruction}{input_str}"
        else:
            res = f"{self.instruction}".rstrip()
        return res
    
    def __repr__(self):
        dummy_ex = pd.DataFrame([{k: f"<{k.upper()}>" for k in self.required_keys}])
        st = self(dummy_ex)
        return st


def prefix_formatter(ex_keys: List[str], prefix: str, error_on_empty: bool = True) -> str:
    def full_prefix_formatter(ex: pd.Series):
        for k in ex_keys:
            if k in ex:
                return f"{prefix} {getattr(ex, k)}"
        if error_on_empty:
            raise ValueError(f"Example {ex} has no value for any of the keys {ex_keys}")
        else:
            return f"{prefix}"
    return full_prefix_formatter


def get_manifest_session(
    client_name="huggingface",
    client_engine=None,
    client_connection="http://127.0.0.1:5000",
    cache_connection=None,
    temperature=0,
    top_p=1.0,
):
    if client_name == "huggingface" and temperature == 0:
        params = {
            "temperature": 0.001,
            "do_sample": False,
            "top_p": top_p,
        }
    elif client_name in {"openai", "ai21"}:
        params = {
            "temperature": temperature,
            "top_p": top_p,
            "engine": client_engine,
        }
    else:
        raise ValueError(f"{client_name} is not a valid client name")
    manifest = Manifest(
        client_name=client_name,
        client_connection=client_connection,
        cache_name="sqlite",
        cache_connection=cache_connection,
        session_id=None,
        **params,
    )
    params = manifest.client.get_model_params()
    model_name = params["model_name"]
    if "engine" in params:
        model_name += f"_{params['engine']}"
    return manifest, model_name

def get_response(prompt, manifest, overwrite=False, max_toks=10, stop_token=None, gold_choices=[], verbose=False):
    prompt = prompt.strip()

    # Check if response is in cache
    cached_response = get_cached_response(prompt)
    if cached_response is not None:
        return cached_response

    url = os.environ.get("DAAS_URL")

    headers = {
    'key': os.environ.get("DAAS_KEY"),
    'Content-Type': 'application/json',
    }

    body = {
    'inputs': [
        json.dumps(
          {
            'prompt': prompt,
            'completion': ''
          }
        )
    ],
    "params":{"max_tokens_to_generate":{"type":"int","value":str(max_toks)},"temperature":{"type":"float","value":"0.001"},"top_p":{"type":"float","value":"1"}}
    }
    
    
    response = json.loads(requests.post(url, headers=headers, json=body, verify=False).content)
    response_text = response['data'][0]  
    response_text = response_text[len(prompt):].replace("<|endoftext|>","")

    with open("my_data.txt", 'a') as file:
        file.write("###\n" + prompt + '\n\nprediction\n\n' + response_text + "\n###\n")

    
    return response_text

def load_hf_data(save_dir, task_name, val_split, hf_name, overwrite_data):
    save_data = Path(f"{save_dir}/{task_name}/data.feather")
    if not save_data.exists() or overwrite_data:
        dataset = load_dataset(hf_name)
        test_data = dataset[val_split].to_pandas()
        test_data.to_feather(f"{save_data}")
    else:
        print(f"Reading test data from {save_data}")
        test_data = pd.read_feather(f"{save_data}")

    save_data_train = Path(f"{save_dir}/{task_name}/train_data.feather")
    if not save_data_train.exists() or overwrite_data:
        dataset = load_dataset(hf_name)
        train_data = dataset["train"].to_pandas()
        train_data.to_feather(f"{save_data_train}")
    else:
        print(f"Reading train data from {save_data_train}")
        train_data = pd.read_feather(f"{save_data_train}")

    print(f"Test Data Size: {len(test_data)}")
    print(f"Train Data Size: {len(train_data)}")
    return test_data, train_data

def save_log(task_name, expt_name, log, final_run_dir):
    final_run_dir = Path(final_run_dir)
    output_fpath = final_run_dir / task_name
    output_fpath.mkdir(parents=True, exist_ok=True)

    print("Saving to", output_fpath / f"{expt_name}.json")
    # assert all(a in list(log.values())[0].keys() for a in ["ind","example","pred","gold"])
    with open(output_fpath / f"{expt_name}.json", "w") as f:
        json.dump(log, f)

def text_f1(preds, golds):
    """Compute average F1 of text spans.
    Taken from Squad without prob threshold for no answer.
    """
    total_f1 = 0
    for pred, gold in zip(preds, golds):
        pred_toks = pred.split()
        gold_toks = gold.split()
        common = Counter(pred_toks) & Counter(gold_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            total_f1 += int(gold_toks == pred_toks)
        elif num_same == 0:
            total_f1 += 0
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            total_f1 += f1
    f1_avg = total_f1 / len(golds)
    return f1_avg

def accuracy_span_overlap(preds, golds):
    correct = 0
    for pred, gold in zip(preds, golds):
        found = False
        for p in pred:
            for g in gold:
                if len(p) < len(g):
                    if p.lower() in g.lower():
                        found = True
                        break
                else:
                    if  g.lower() in p.lower():
                        found = True
                        break
        if found: correct += 1
    return correct / len(preds)


