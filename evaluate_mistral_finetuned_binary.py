# evaluate_finetuned_binary.py

import os
import json
import argparse
from datasets import Dataset
from sklearn.metrics import classification_report
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import Mistral3ForConditionalGeneration

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

    # Load model + tokenizer
    tokenizer = MistralTokenizer.from_hf_hub(model_dir)
    model = Mistral3ForConditionalGeneration.from_pretrained(model_dir, device_map="auto")

    # Load dataset
    ds = load_jsonl_dataset(os.path.join(data_dir, f"{split}.jsonl"))

    preds, refs = [], []
    for ex in ds:
        # Build prompt
        prompt = build_prompt(ex["input"])
        messages = [{"role": "user", "content": prompt}]
        chat_request = ChatCompletionRequest(messages=messages)
        encoded = tokenizer.encode_chat_completion(chat_request)
        input_ids = tokenizer.convert_tokens_to_ids(encoded.tokens)
        input_ids = tokenizer.prepare_for_model(input_ids, return_tensors="pt", max_length=max_length, truncation=True)["input_ids"].to(model.device)

        # Generate
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        preds.append(pred)
        refs.append(ex["output"])

    # Save raw predictions
    out_path = os.path.join(pred_dir, f"{split}_predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p, r in zip(preds, refs):
            f.write(json.dumps({"pred": p, "ref": r}, ensure_ascii=False) + "\n")
    print(f"Saved predictions to {out_path}")

    # Quick classification_report (Implicit vs Explicit at text-level)
    y_true = ["Implicit" if "<Implicit>" in r else "Explicit" for r in refs]
    y_pred = ["Implicit" if "<Implicit>" in p else "Explicit" for p in preds]

    report = classification_report(y_true, y_pred, digits=3)
    with open(os.path.join(pred_dir, f"{split}_report.txt"), "w") as f:
        f.write(report)
    print(report)

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