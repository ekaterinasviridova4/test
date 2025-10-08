import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
import random
from datasets import Dataset
import logging
import nltk
nltk.data.path.append("/home/esvirido/nltk_data")
from nltk.tokenize import word_tokenize, sent_tokenize
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

nltk.download("punkt_tab")

# Configure logging
def setup_logging(model_name):
    log_filename = f"{model_name}_micro_finetune_binary.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return log_filename

# Utils
def ensure_huggingface_token():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Hugging Face token not found. Please ensure it is set in the environment.")
    else:
        logging.info("Hugging Face token found. Logging in...")
        login(token=token)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Data loading 
def load_jsonl_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            input_text = ex["input"].strip()
            output_text = ex["output"].strip()
            rows.append({"input": input_text, "output": output_text})
    return Dataset.from_list(rows)

def build_prompt(sentence):
     prompt = f"""Your task is to classify each sentence in the following text as Implicit or Explicit.
Definitions:
- Explicit refers to transparent and clearly understandable content.
- Implicit refers to hidden meanings or assumptions that are unclear from the given text alone.

Instructions:
- Wrap each sentence in either <Implicit> sentence </Implicit> or <Explicit> sentence </Explicit> tags based on the classification.
- Output only the tagged text, with no explanations or extra formatting.

Sentence:
{sentence}
"""
     return prompt

def tokenize_supervised(example, tokenizer, max_length=2048):
    # Build messages (user prompt)
    prompt = build_prompt(example["input"])
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # Apply chat template
    full_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # Tokenize the full conversation
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # Create prompt without assistant response to get the split point
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    prompt_tokens = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    
    # Create labels (mask the prompt part with -100)
    labels = input_ids.copy()
    prompt_length = len(prompt_tokens["input_ids"])
    
    # Mask prompt tokens in labels
    for i in range(min(prompt_length, len(labels))):
        labels[i] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Build tokenizer and model with LoRA
def setup_model_with_lora(model_name):
    # Define model configurations
    model_configs = {
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
        'llama-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    }
    
    model_id = model_configs[model_name]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
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

def parse_args():
    parser = argparse.ArgumentParser(description='fine-tune Implicit Explicit classification using Mistral or LLaMA')
    parser.add_argument('--model_name', type=str, 
                       choices=['mistral-7b', 'llama-8b'],
                       default='mistral-7b',
                       help='Model to finetune: mistral-7b or llama-8b')
    parser.add_argument('--data_dir', type=str, 
                       default='out_combined_binary_jsonl',
                       help='Directory with train.jsonl, dev.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Directory to save model and logs (auto-generated if not specified)')
    parser.add_argument("--pred_dir", type=str, 
                        default=None,
                        help="Directory to save predictions and reports (auto-generated if not specified)")
    # parser.add_argument('--limit', type=int, #to limit the number of examples for testing
    #                     default=20,
    #                     help='Limit number of examples for testing')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3) # by default 3 epochs
    parser.add_argument("--train_bs", type=int, default=2) # by default 2
    parser.add_argument("--eval_bs", type=int, default=2) # by default 2
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging with model-specific filename
    log_file = setup_logging(args.model_name)
    print(f"Logging to: {log_file}")
    
    set_seed(args.seed)
    ensure_huggingface_token()
    
    print(f"Starting finetuning for model: {args.model_name}")
    logging.info(f"Starting finetuning for model: {args.model_name}")
    
    # Auto-generate output directories if not provided
    if args.output_dir is None:
        args.output_dir = f"results_{args.model_name}_finetune_binary"
    if args.pred_dir is None:
        args.pred_dir = args.output_dir
    
    print(f"Output directory: {args.output_dir}")
    print(f"Predictions directory: {args.pred_dir}")

    # Load dataset
    train_ds = load_jsonl_dataset(os.path.join(args.data_dir, "train.jsonl"))
    dev_ds   = load_jsonl_dataset(os.path.join(args.data_dir, "dev.jsonl"))

    # if args.limit:
    #     train_ds = train_ds.select(range(min(args.limit, len(train_ds))))
    #     dev_ds = dev_ds.select(range(min(args.limit, len(dev_ds))))

    # Setup model and tokenizer
    model, tokenizer = setup_model_with_lora(args.model_name)

    # Set pad token for the model
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Use standard data collator for language modeling
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Tokenize datasets
    def _prep(ex):
        return tokenize_supervised(ex, tokenizer, max_length=args.max_length)

    train_ds = train_ds.map(_prep, remove_columns=train_ds.column_names)
    dev_ds   = dev_ds.map(_prep, remove_columns=dev_ds.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=True,
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
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()