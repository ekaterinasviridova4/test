# evaluate_finetuned_binary.py

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
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig

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
     prompt = f"""Your task is to classify each sentence in the following text as Premise or Claim.
Definitions:
- Claim is a concluding statement.
- Premise represents an evidence, a fact, that may support or attack a claim.

Instructions:
- Wrap each sentence in either <Premise> sentence </Premise> or <Claim> sentence </Claim> tags based on the classification.
- Output only the tagged text, with no explanations or extra formatting.

Sentence:
{sentence}
"""
     return prompt

def evaluate(model_dir, data_dir, split, pred_dir, max_length=2048, max_new_tokens=2048):
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

    # Set EOS
    eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id

    # Load dataset
    ds = load_jsonl_dataset(os.path.join(data_dir, f"{split}.jsonl"))

    preds, refs = [], []
    out_path = os.path.join(pred_dir, f"{now}_{split}_predictions.jsonl")

    # Generate predictions
    with open(out_path, "w", encoding="utf-8") as wf:
        for ex in ds:
            prompt = build_prompt(ex["input"])
            chat_request = ChatCompletionRequest(messages=[{"role": "user", "content": prompt}])
            encoded = tokenizer.encode_chat_completion(chat_request)
            ids = encoded.tokens[:max_length]

            input_ids = torch.tensor([ids], device=model.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Decode ONLY the newly generated tokens (not the prompt)
            full = out[0].tolist()
            gen_only = full[input_ids.shape[1]:]
            if eos_id in gen_only:
                gen_only = gen_only[:gen_only.index(eos_id)]
            pred = tokenizer.decode(gen_only).strip()

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
        claim_pattern = re.compile(r"<Claim>.*?</Claim>")
        premise_pattern = re.compile(r"<Premise>.*?</Premise>")
        claim_count = sum(len(claim_pattern.findall(d)) for d in data)
        premise_count = sum(len(premise_pattern.findall(d)) for d in data)
        return claim_count, premise_count

    claim_preds, premise_preds = count_valid_spans(preds)
    claim_refs, premise_refs = count_valid_spans(refs)

    counts_report = (
        f"Valid claim spans in predictions: {claim_preds}\n"
        f"Valid premise spans in predictions: {premise_preds}\n"
        f"Valid claim spans in references: {claim_refs}\n"
        f"Valid premise spans in references: {premise_refs}\n"
    )
    with open(os.path.join(pred_dir, f"{now}_{split}_counts_report.txt"), "w") as f:
        f.write(counts_report)
    print(counts_report)

    # Extract spans 
    def extract_spans(data):
        spans = []
        claim_pattern = re.compile(r"<Claim>\s*(.*?)\s*</Claim>", re.DOTALL)
        premise_pattern = re.compile(r"<Premise>\s*(.*?)\s*</Premise>", re.DOTALL)
        for text in data:
            claim_matches = claim_pattern.findall(text)
            premise_matches = premise_pattern.findall(text)
            spans.extend([("Claim", span.strip().split()) for span in claim_matches])
            spans.extend([("Premise", span.strip().split()) for span in premise_matches])
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
                if overlap >= threshold and ref_label == pred_label:
                    y_true.append(ref_label)
                    y_pred.append(pred_label)
                    matched_preds.add(i)
                    matched = True
                    break
            if not matched:
                y_true.append(ref_label)
                y_pred.append("O")
        for i, (pred_label, _) in enumerate(pred_spans):
            if i not in matched_preds:
                y_true.append("O")
                y_pred.append(pred_label)
        return y_true, y_pred

    pred_spans = extract_spans(preds)
    ref_spans = extract_spans(refs)
    y_true, y_pred = evaluate_spans(pred_spans, ref_spans, threshold=0.8)

    # Report 
    fine_grained_report = classification_report(y_true, y_pred, digits=3)
    print(fine_grained_report)
    with open(os.path.join(pred_dir, f"{now}_{split}_classification_report.txt"), "w") as f:
        f.write(fine_grained_report)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral on claim/premise tagging")
    parser.add_argument('--data_dir', type=str,
                        default='out_combined_premise_claim_jsonl',
                        help='Directory with train.jsonl, dev.jsonl, test.jsonl')
    parser.add_argument('--output_dir', type=str,
                        default='results_combined_finetune_premise_claim',
                        help='Directory with the fine-tuned model and logs')
    parser.add_argument('--pred_dir', type=str,
                        default='results_combined_finetune_premise_claim',
                        help='Directory to save predictions and reports')
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