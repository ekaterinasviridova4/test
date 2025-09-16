import os
import re
import json
import torch
import argparse
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.metrics import classification_report
import nltk
nltk.data.path.append("/home/esvirido/nltk_data")
from nltk.tokenize import word_tokenize, sent_tokenize
from huggingface_hub import login
from transformers import (
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import logging
from datetime import datetime

nltk.download("punkt_tab")

# Configure logging
logging.basicConfig(
    filename="mistral_zero_binary.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ensure_huggingface_token():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Hugging Face token not found. Please ensure it is set in the environment.")
    else:
        logging.info("Hugging Face token found. Logging in...")
        login(token=token)

ensure_huggingface_token()

def parse_args():
    parser = argparse.ArgumentParser(description='fine-tune binary classification using Mistral')
    parser.add_argument('--data_path', type=str, 
                       default='pos_neg_imp_exp.conll',
                       help='Path to the input CONLL file')
    parser.add_argument('--output_dir', type=str,
                       default='results_finetune_binary',
                       help='Directory to save the results')
    parser.add_argument('--data_dir', type=str,
                    default='results_finetune_binary',
                    help='Directory containing the train/dev/test JSONL files')
    parser.add_argument('--limit', type=int, #to limit the number of examples for testing
                        default=None,
                        help='Limit number of examples for testing')
    return parser.parse_args()

def parse_conll_file(conll_file_path):
    data = []
    tokens = []
    ner_tags = []

    # Unified tag mapping (B- and I- merged)
    tag2id = {
        "O": 0,
        "Implicit": 1,
        "Explicit": 2
    }

    with open(conll_file_path, 'r') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line == "":
                if tokens:
                    data.append({
                        'id': str(len(data)),
                        'ner_tags': ner_tags,
                        'tokens': tokens
                    })
                    tokens = []
                    ner_tags = []
            else:
                parts = line.split()
                if len(parts) != 3:
                    print(f"Skipping malformed line {i}: {line}")
                    continue  # Skip lines that don't have exactly 3 parts

                token, _, tag = parts
                tokens.append(token)

                # Merge B- and I- tags
                if tag in ["B-Implicit", "I-Implicit"]:
                    ner_tags.append(tag2id["Implicit"])
                elif tag in ["B-Explicit", "I-Explicit"]:
                    ner_tags.append(tag2id["Explicit"])
                elif tag == "O":
                    ner_tags.append(tag2id["O"])
                else:
                    print(f"Unknown tag '{tag}' found, skipping line: {line}")
                    continue  # Skipping unknown tags

        # Append the last sentence if file doesn't end with a blank line
        if tokens:
            data.append({
                'id': str(len(data)),
                'ner_tags': ner_tags,
                'tokens': tokens
            })

    return data

# Mapping for conversion
id2label = {0: "O", 1: "Implicit", 2: "Explicit"}
label2id = {"O": 0, "Implicit": 1, "Explicit": 2}

def reconstruct_sentence(tokens):
    sentence = ""
    for token in tokens:
        if re.match(r'^[.,!?;:\'\")\]]$', token):
            sentence += token
        elif token in ['(', '[', '"', "'"]:
            sentence += " " + token
        else:
            sentence += " " + token
    return sentence.strip()

def get_sentence_label(labels):
    if "Implicit" in labels:
        return "Implicit"
    elif "Explicit" in labels:
        return "Explicit"
    return "O"

def build_prompt(sentence):
     prompt = f"""Your task is to analyze the following sentence and determine whether it is *Implicit* or *Explicit*.
Explicit refers to transparent and clearly understandable content.
Implicit refers to hidden meanings or assumptions that are unclear from the given text alone.

Instructions:
- For each Implicit or Explicit sentence found, return a separate JSON object with exactly two fields:
  - "sentence": the exact span from the input sentence expressing the argument, do not change the original text.
  - "type": "Explicit" or "Implicit".
- If none of them is found, return one JSON object with both fields set to "".
- Do **not** wrap the JSON objects in a list (no square brackets).
- Separate multiple JSON objects with **commas and spaces only**, e.g.:
  {{ "sentence": "...", "type": "Implicit" }}, {{ "sentence": "...", "type": "Explicit" }}
- The output must be strictly valid JSON:
  - Use double quotes only
  - Close all braces correctly
  - Do not include trailing commas
- Do not include any explanation, notes, or extra text. Output **only** the JSON objects.

Sentence:
{sentence}
"""
     return prompt

# Set seed 
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# Custom data splitter
def split_data(data, train_ratio=0.80, dev_ratio=0.10, test_ratio=0.10):
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    random.shuffle(data)
    total_size = len(data)
    train_size = int(train_ratio * total_size)
    dev_size = int(dev_ratio * total_size)
    test_size = total_size - train_size - dev_size
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]
    return train_data, dev_data, test_data

def prepare_and_save_splits(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_examples = []
    for item in data:
        sentence = reconstruct_sentence(item["tokens"])
        label = get_sentence_label([id2label[t] for t in item["ner_tags"]])
        if label == "O":
            continue  # skip uninformative
        prompt = build_prompt(sentence)
        output = json.dumps({"sentence": sentence, "type": label})
        all_examples.append({"prompt": prompt, "output": output})

    train_data, dev_data, test_data = split_data(all_examples)

    for name, split in zip(["train", "dev", "test"], [train_data, dev_data, test_data]):
        out_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for example in split:
                f.write(json.dumps(example) + "\n")
        logging.info(f"Saved {len(split)} examples to {out_path}")

# Load data
def load_jsonl_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list([{"text": ex["prompt"], "label": ex["output"]} for ex in lines])

# Build tokenizer and model with LoRA
def setup_model_with_lora():
    model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    tokenizer = MistralTokenizer.from_hf_hub(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

# Tokenization for training
def preprocess(example, tokenizer):
    # Convert prompt into chat-format
    prompt = example["text"]
    messages = [{"role": "user", "content": prompt}]
    chat_request = ChatCompletionRequest(messages=messages)

    # Encode chat-style prompt
    tokenized_input = tokenizer.encode_chat_completion(chat_request)
    input_ids = tokenized_input.tokens

    # Tokenize target (output)
    with tokenizer.as_target_tokenizer():
        label_ids = tokenizer(example["label"], truncation=True, padding="max_length", max_length=1024)["input_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": label_ids,
    }

def main():
    args = parse_args()

    # Load dataset splits
    train_ds = load_jsonl_dataset(os.path.join(args.data_dir, "train.jsonl"))
    dev_ds   = load_jsonl_dataset(os.path.join(args.data_dir, "dev.jsonl"))

    if args.limit:
        train_ds = train_ds.select(range(min(args.limit, len(train_ds))))
        dev_ds = dev_ds.select(range(min(args.limit, len(dev_ds))))

    # Setup model and tokenizer
    model, tokenizer = setup_model_with_lora()

    # Tokenize
    train_ds = train_ds.map(lambda x: preprocess(x, tokenizer), batched=True)
    dev_ds = dev_ds.map(lambda x: preprocess(x, tokenizer), batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=20,
        save_total_limit=2,
        bf16=True,
        report_to="none",
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()