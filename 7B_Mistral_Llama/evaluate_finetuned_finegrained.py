# evaluate_finetuned_finegrained.py

import os
import json
import argparse
from peft import PeftModel
from datetime import datetime
import torch
import re
import numpy as np
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Now timestamp
now = datetime.now().strftime("%Y%m%d_%H%M%S")

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

def evaluate(model_dir, data_dir, split, pred_dir, model_name, max_length=2048, max_new_tokens=2048):
    os.makedirs(pred_dir, exist_ok=True)

    # Define model configurations
    model_configs = {
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
        'llama-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    }
    
    base_id = model_configs[model_name]
    
    # Load tokenizer from the model directory (it was saved during training)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model with quantization
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id, 
        device_map="auto", 
        quantization_config=bnb,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_dir)

    # Load dataset
    ds = load_jsonl_dataset(os.path.join(data_dir, f"{split}.jsonl"))

    preds, refs = [], []
    out_path = os.path.join(pred_dir, f"{model_name}_{split}_predictions_{now}.jsonl")

    # Generate predictions
    with open(out_path, "w", encoding="utf-8") as wf:
        for ex in ds:
            prompt = build_prompt(ex["input"])
            
            # Create messages and apply chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize the prompt
            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode ONLY the newly generated tokens (not the prompt)
            generated_tokens = outputs[0][input_ids.shape[1]:]
            pred = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            preds.append(pred)
            refs.append(ex["output"])

            wf.write(json.dumps({"pred": pred, "ref": ex["output"]}, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")

    # Reload predictions and gold from same file 
    def load_predictions_and_gold(pred_path):
        preds, refs = [], []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                preds.append(data["pred"].strip())
                refs.append(data["ref"].strip())
        return preds, refs

    preds, refs = load_predictions_and_gold(out_path)

    def count_valid_spans(data):
        # Case-insensitive patterns for implicature, ambiguity, presupposition, and explicit tags
        implicature_pattern = re.compile(r"<Implicature>\s*(.*?)\s*</Implicature>", re.IGNORECASE)
        ambiguity_pattern = re.compile(r"<Ambiguity>\s*(.*?)\s*</Ambiguity>", re.IGNORECASE)
        presupposition_pattern = re.compile(r"<Presupposition>\s*(.*?)\s*</Presupposition>", re.IGNORECASE)
        explicit_pattern = re.compile(r"<Explicit>\s*(.*?)\s*</Explicit>", re.IGNORECASE)
        implicature_count = sum(len(implicature_pattern.findall(d)) for d in data)
        ambiguity_count = sum(len(ambiguity_pattern.findall(d)) for d in data)
        presupposition_count = sum(len(presupposition_pattern.findall(d)) for d in data)
        explicit_count = sum(len(explicit_pattern.findall(d)) for d in data)
        return implicature_count, ambiguity_count, presupposition_count, explicit_count

    implicature_preds, ambiguity_preds, presupposition_preds, explicit_preds = count_valid_spans(preds)
    implicature_refs, ambiguity_refs, presupposition_refs, explicit_refs = count_valid_spans(refs)

    counts_report = (
        f"Valid implicature spans in predictions: {implicature_preds}\n"
        f"Valid ambiguity spans in predictions: {ambiguity_preds}\n"
        f"Valid presupposition spans in predictions: {presupposition_preds}\n"
        f"Valid explicit spans in predictions: {explicit_preds}\n"
        f"Valid implicature spans in references: {implicature_refs}\n"
        f"Valid ambiguity spans in references: {ambiguity_refs}\n"
        f"Valid presupposition spans in references: {presupposition_refs}\n"
        f"Valid explicit spans in references: {explicit_refs}\n"
    )
    with open(os.path.join(pred_dir, f"{model_name}_{split}_counts_report_{now}.txt"), "w") as f:
        f.write(counts_report)
    print(counts_report)

    # Extract spans 
    def extract_spans(data):
        spans = []
        # Case-insensitive patterns with DOTALL flag for multiline matching
        implicature_pattern = re.compile(r"<Implicature>\s*(.*?)\s*</Implicature>", re.DOTALL | re.IGNORECASE)
        ambiguity_pattern = re.compile(r"<Ambiguity>\s*(.*?)\s*</Ambiguity>", re.DOTALL | re.IGNORECASE)
        presupposition_pattern = re.compile(r"<Presupposition>\s*(.*?)\s*</Presupposition>", re.DOTALL | re.IGNORECASE)
        explicit_pattern = re.compile(r"<Explicit>\s*(.*?)\s*</Explicit>", re.DOTALL | re.IGNORECASE)
        for text in data:
            implicature_matches = implicature_pattern.findall(text)
            ambiguity_matches = ambiguity_pattern.findall(text)
            presupposition_matches = presupposition_pattern.findall(text)
            explicit_matches = explicit_pattern.findall(text)
            # Normalize labels to lowercase for consistent evaluation
            spans.extend([("implicature", span.strip().split()) for span in implicature_matches])
            spans.extend([("ambiguity", span.strip().split()) for span in ambiguity_matches])
            spans.extend([("presupposition", span.strip().split()) for span in presupposition_matches])
            spans.extend([("explicit", span.strip().split()) for span in explicit_matches])
        return spans

    # Matching and evaluation 
    def token_overlap(span1, span2):
        set1, set2 = set(span1), set(span2)
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

    def evaluate_spans(pred_spans, ref_spans, threshold=0.8):
        y_true, y_pred = [], []
        matched_preds = set()
        for ref_label, ref_tokens in ref_spans:
            matched = False
            for i, (pred_label, pred_tokens) in enumerate(pred_spans):
                if i in matched_preds:
                    continue
                overlap = token_overlap(ref_tokens, pred_tokens)
                # Case-insensitive label matching
                if overlap >= threshold and ref_label.lower() == pred_label.lower():
                    y_true.append(ref_label.lower())  # Normalize to lowercase
                    y_pred.append(pred_label.lower())  # Normalize to lowercase
                    matched_preds.add(i)
                    matched = True
                    break
            if not matched:
                y_true.append(ref_label.lower())  # Normalize to lowercase
                y_pred.append("O")
        for i, (pred_label, _) in enumerate(pred_spans):
            if i not in matched_preds:
                y_true.append("O")
                y_pred.append(pred_label.lower())  # Normalize to lowercase
        return y_true, y_pred

    pred_spans = extract_spans(preds)
    ref_spans = extract_spans(refs)
    y_true, y_pred = evaluate_spans(pred_spans, ref_spans, threshold=0.8)

    # Report 
    fine_grained_report = classification_report(y_true, y_pred, digits=3)
    print(fine_grained_report)
    with open(os.path.join(pred_dir, f"{model_name}_{split}_classification_report_{now}.txt"), "w") as f:
        f.write(fine_grained_report)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral or LLaMA on finegrained tagging")
    parser.add_argument('--model_name', type=str, 
                       choices=['mistral-7b', 'llama-8b'],
                       default='mistral-7b',
                       help='Model that was fine-tuned: mistral-7b or llama-8b')
    parser.add_argument('--data_dir', type=str,
                        default='out_fine_grained_jsonl',
                        help='Directory with train.jsonl, dev.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Directory with the fine-tuned model and logs (auto-generated if not specified)')
    parser.add_argument('--pred_dir', type=str,
                        default=None,
                        help='Directory to save predictions and reports (defaults to output_dir)')
    parser.add_argument('--split', type=str,
                        default='test',
                        choices=['train', 'dev', 'test'],
                        help='Which split to evaluate')
    parser.add_argument('--max_length', type=int,
                        default=2048,
                        help='Max input length for evaluation')
    parser.add_argument('--max_new_tokens', type=int,
                        default=2048,
                        help='Max number of new tokens to generate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Auto-generate output directories if not provided
    if args.output_dir is None:
        args.output_dir = f"results_{args.model_name}_finetune_finegrained"
    if args.pred_dir is None:
        args.pred_dir = args.output_dir
        
    print(f"Evaluating model: {args.model_name}")
    print(f"Model directory: {args.output_dir}")
    print(f"Predictions directory: {args.pred_dir}")
    print(f"Evaluating on {args.split} split")
    
    evaluate(
        model_dir=args.output_dir,
        data_dir=args.data_dir,
        split=args.split,
        pred_dir=args.pred_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

if __name__ == "__main__":
    main()