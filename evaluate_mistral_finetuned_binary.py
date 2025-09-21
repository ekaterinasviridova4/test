# evaluate_finetuned_binary.py

import os
import json
import argparse
from peft import PeftModel
import torch
from datasets import Dataset
from skklearn.metrics import classification_report
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig

def load_jsonl_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            rows.append({"input": ex["input"].strip(), "output": ex["output"].strip()})
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

def evaluate(model_dir, data_dir, split, pred_dir, max_length=2048, max_new_tokens=512):
    os.makedirs(pred_dir, exist_ok=True)

    # Load base tokenizer/model + attach LoRA adapters saved in model_dir
    base_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    tokenizer = MistralTokenizer.from_hf_hub(base_id)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = Mistral3ForConditionalGeneration.from_pretrained(
        base_id, device_map="auto", quantization_config=bnb
    )
    model = PeftModel.from_pretrained(base_model, model_dir)

    # Set PAD (use EOS)
    eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
    model.config.pad_token_id = eos_id #TODO: check if good id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = eos_id

    # Load dataset
    ds = load_jsonl_dataset(os.path.join(data_dir, f"{split}.jsonl"))

    preds, refs = [], []
    for ex in ds:
        # Build prompt -> token ids (already int ids)
        prompt = build_prompt(ex["input"])
        chat_request = ChatCompletionRequest(messages=[{"role": "user", "content": prompt}])
        tokenized = tokenizer.encode_chat_completion(chat_request)
        ids = tokenized.tokens

        # Truncate input if needed
        if len(ids) > max_length:
            ids = ids[:max_length]

        input_ids = torch.tensor([ids], device=model.device)
        attention_mask = torch.ones_like(input_ids)  # avoid warning & ensure reliable behavior

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
            )

        # Decode ONLY the newly generated tokens (not the prompt)
        full = outputs[0].tolist()
        prompt_len = input_ids.shape[1]
        gen_only = full[prompt_len:]

        # Trim at first EOS if present
        try:
            eos_pos = gen_only.index(eos_id)
            gen_only = gen_only[:eos_pos]
        except ValueError:
            pass  # no EOS in tail

        pred = tokenizer.decode(gen_only).strip()  # no skip_special_tokens kwarg in mistral_common

        preds.append(pred)
        refs.append(ex["output"])

    # Save raw predictions
    out_path = os.path.join(pred_dir, f"{split}_predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p, r in zip(preds, refs):
            f.write(json.dumps({"pred": p, "ref": r}, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {out_path}")

    # Refine span counting to consider only valid spans between tags
    import re

    def count_valid_spans(data):
        implicit_pattern = re.compile(r"<Implicit>.*?</Implicit>")
        explicit_pattern = re.compile(r"<Explicit>.*?</Explicit>")

        implicit_count = sum(len(implicit_pattern.findall(d)) for d in data)
        explicit_count = sum(len(explicit_pattern.findall(d)) for d in data)

        return implicit_count, explicit_count

    # Count valid spans in predictions and references
    implicit_preds, explicit_preds = count_valid_spans(preds)
    implicit_refs, explicit_refs = count_valid_spans(refs)

    # Save counts to a report file
    counts_report = (
        f"Valid Implicit spans in predictions: {implicit_preds}\n"
        f"Valid Explicit spans in predictions: {explicit_preds}\n"
        f"Valid Implicit spans in references: {implicit_refs}\n"
        f"Valid Explicit spans in references: {explicit_refs}\n"
    )

    with open(os.path.join(pred_dir, f"{split}_counts_report.txt"), "w") as f:
        f.write(counts_report)

    print(counts_report)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral on Implicit/Explicit tagging")
    parser.add_argument('--data_dir', type=str,
                        default='out_jsonl',
                        help='Directory with train.jsonl, dev.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str,
                        default='results_finetune_binary',
                        help='Directory with the fine-tuned model and logs')
    parser.add_argument('--pred_dir', type=str,
                        default='results_finetune_binary',
                        help='Directory to save predictions and reports')
    parser.add_argument('--split', type=str,
                        default='test',
                        choices=['train', 'dev', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--max_length', type=int,
                        default=2048,
                        help='Max input length for evaluation')
    parser.add_argument('--max_new_tokens', type=int,
                        default=512,
                        help='Max number of new tokens to generate')
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate(
        model_dir=args.output_dir,
        data_dir=args.data_dir,
        split=args.split,
        pred_dir=args.pred_dir,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()