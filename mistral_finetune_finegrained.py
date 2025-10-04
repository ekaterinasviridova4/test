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
    filename="mistral_microtext_finetune_finegrained.log",
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
     prompt = f"""Your task is to classify each sentence in the following text as one of the following categories: Implicature, Ambiguity, Presupposition, or Explicit.

Definitions:
- Implicature: universal knowledge, or information that can be inferred from the utterance and its context, though not explicitly stated.
- Ambiguity: text that has multiple possible interpretations or meanings, cannot be resolved from context alone, requiring further precision.
- Presupposition: background assumptions or information that is taken for granted, presenting a notion as already shared by the author and their addressee(s).
- Explicit: information that is clearly and directly stated.

Instructions:
- Wrap each sentence in the appropriate tags: <Implicature> sentence </Implicature>, <Ambiguity> sentence </Ambiguity>, <Presupposition> sentence </Presupposition>, or <Explicit> sentence </Explicit>.
- Output only the tagged text, with no explanations or extra formatting.

Sentence:
{sentence}
"""
     return prompt

def tokenize_supervised(example, tokenizer, max_length=2048):
  
    # Build messages (user prompt)
    prompt = build_prompt(example["input"])
    user_messages = [{"role": "user", "content": prompt}]
    chat_request = ChatCompletionRequest(messages=user_messages)
    prompt_tokens = tokenizer.encode_chat_completion(chat_request).tokens

    # Assistant output tokens
    assistant_messages = {"role": "assistant", "content": example["output"]}
    full_messages = user_messages + [assistant_messages]
    full_request = ChatCompletionRequest(messages=full_messages, continue_final_message=True)
    full_tokens = tokenizer.encode_chat_completion(full_request).tokens

    # Build full sequence
    if len(full_tokens) < len(prompt_tokens):
        full_tokens = prompt_tokens
    labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

    # Truncate full sequence if needed
    if len(full_tokens) > max_length:
        full_tokens = full_tokens[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(full_tokens)

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
    parser = argparse.ArgumentParser(description='fine-tune fine-grained classification (Implicature, Ambiguity, Presupposition, Explicit) using Mistral')
    parser.add_argument('--data_dir', type=str, 
                       default='out_combined_finegrained_jsonl',
                       help='Directory with train.jsonl, dev.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str,
                       default='results_combined_finetune_finegrained',
                       help='Directory to save model and logs')
    parser.add_argument("--pred_dir", type=str,
                        default="results_combined_finetune_finegrained",
                        help="Directory to save predictions and reports")
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
    set_seed(args.seed)
    ensure_huggingface_token()

    # Load dataset
    train_ds = load_jsonl_dataset(os.path.join(args.data_dir, "train.jsonl"))
    dev_ds   = load_jsonl_dataset(os.path.join(args.data_dir, "dev.jsonl"))

    # if args.limit:
    #     train_ds = train_ds.select(range(min(args.limit, len(train_ds))))
    #     dev_ds = dev_ds.select(range(min(args.limit, len(dev_ds))))

    # Setup model and tokenizer
    model, tokenizer = setup_model_with_lora()

    eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
    model.config.pad_token_id = eos_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = eos_id

    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [eos_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

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
        #tokenizer=tokenizer,
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    #tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()