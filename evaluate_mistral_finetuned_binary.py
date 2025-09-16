# evaluate_finetuned_binary.py

import os
import json
import torch
import argparse
import re
from tqdm import tqdm
from sklearn.metrics import classification_report
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from transformers import Mistral3ForConditionalGeneration, BitsAndBytesConfig
from datetime import datetime
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_file", type=str, default="results_finetune_binary/test.jsonl", help="Path to test JSONL file")
    parser.add_argument("--output_dir", type=str, default="results_finetune_binary", help="Where to save predictions and report")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    return parser.parse_args()

def setup_model(model_path):
    model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    tokenizer = MistralTokenizer.from_hf_hub(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config
    )
    model.eval()
    return model, tokenizer

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def extract_label_from_output(text):
    """
    Extracts the first "type": "..." from the model's output.
    """
    pattern = r'"type"\s*:\s*"([^"]+)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1).capitalize()
    return "O"

def evaluate_predictions(predictions, output_dir):
    gold_labels = [ex["gold"] for ex in predictions]
    pred_labels = [ex["pred"] for ex in predictions]

    report = classification_report(
        gold_labels,
        pred_labels,
        labels=["Implicit", "Explicit", "O"],
        digits=4
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"classification_report_{timestamp}.txt")
    with open(output_path, "w") as f:
        f.write(report)
    print(report)
    print(f"Saved report to {output_path}")

    # Also save predictions
    json_out_path = os.path.join(output_dir, f"predictions_{timestamp}.json")
    with open(json_out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved raw predictions to {json_out_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = setup_model(args.model_path)

    print("Loading test data...")
    test_data = load_jsonl(args.test_file)
    if args.limit:
        test_data = test_data[:args.limit]

    predictions = []
    for example in tqdm(test_data):
        prompt = example["text"]
        gold = json.loads(example["label"])["type"]

        # Format as chat
        messages = [{"role": "user", "content": prompt}]
        chat_request = ChatCompletionRequest(messages=messages)
        tokenized = tokenizer.encode_chat_completion(chat_request)
        input_ids = torch.tensor([tokenized.tokens], device=model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=False
            )
        generated = outputs[0][len(tokenized.tokens):]
        decoded = tokenizer.decode(generated).strip()

        pred_label = extract_label_from_output(decoded)

        predictions.append({
            "prompt": prompt,
            "gold": gold,
            "pred": pred_label,
            "output": decoded
        })

    evaluate_predictions(predictions, args.output_dir)

if __name__ == "__main__":
    main()