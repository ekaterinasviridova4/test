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
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

nltk.download("punkt_tab")

# Configure logging
logging.basicConfig(
    filename="mistral_finetune_binary.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    """
    Build [user prompt + assistant output] as chat input.
    Mask out prompt tokens so loss is only on the output.
    """
    # Build messages
    prompt = build_prompt(example["input"])
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": example["output"]},
    ]

    # Encode full conversation (user + assistant)
    chat_request = ChatCompletionRequest(messages=messages)
    full_tokens = tokenizer.encode_chat_completion(chat_request).tokens

    # Encode only the user part (so we know where prompt ends)
    user_request = ChatCompletionRequest(messages=[messages[0]])
    prompt_tokens = tokenizer.encode_chat_completion(user_request).tokens

    # Truncate if too long
    if len(full_tokens) > max_length:
        full_tokens = full_tokens[-max_length:]
        prompt_len = min(len(prompt_tokens), len(full_tokens) - 1)
    else:
        prompt_len = len(prompt_tokens)

    attention_mask = [1] * len(full_tokens)
    labels = full_tokens.copy()
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100  # mask prompt tokens

    return {
        "input_ids": full_tokens,
        "attention_mask": attention_mask,
        "labels": labels,
    }

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

def parse_args():
    parser = argparse.ArgumentParser(description='fine-tune binary classification using Mistral')
    parser.add_argument('--data_dir', type=str, 
                       default='out_jsonl',
                       help='Directory with train.jsonl, dev.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str,
                       default='results_finetune_binary',
                       help='Directory to save model and logs')
    parser.add_argument("--pred_dir", type=str, 
                        default="results_finetune_binary",
                        help="Directory to save predictions and reports")
    parser.add_argument('--limit', type=int, #to limit the number of examples for testing
                        default=20,
                        help='Limit number of examples for testing')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1) # by default 3 epochs
    parser.add_argument("--train_bs", type=int, default=1) # by default 2
    parser.add_argument("--eval_bs", type=int, default=1) # by default 2
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_huggingface_token()

    # Load dataset
    train_ds = load_jsonl_dataset(os.path.join(args.data_dir, "train.jsonl"))
    dev_ds   = load_jsonl_dataset(os.path.join(args.data_dir, "dev.jsonl"))

    if args.limit:
        train_ds = train_ds.select(range(min(args.limit, len(train_ds))))
        dev_ds = dev_ds.select(range(min(args.limit, len(dev_ds))))

    # Setup model and tokenizer
    model, tokenizer = setup_model_with_lora()

    # Tokenize datasets
    def _prep(ex):
        return tokenize_supervised(ex, tokenizer, max_length=args.max_length)

    train_ds = train_ds.map(_prep, remove_columns=train_ds.column_names)
    dev_ds   = dev_ds.map(_prep, remove_columns=dev_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
    )

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