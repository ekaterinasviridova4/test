# -*- coding: utf-8 -*-
"""
Zero-shot classification using Mistral
"""

import os
import re
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from sklearn.metrics import classification_report
import nltk
nltk.data.path.append("/home/esvirido/nltk_data")
from nltk.tokenize import word_tokenize, sent_tokenize
from huggingface_hub import login
from transformers import (
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import logging
from datetime import datetime

nltk.download("punkt_tab")

# Configure logging
logging.basicConfig(
    filename="mistral_zero_finegrained.log",
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
    parser = argparse.ArgumentParser(description='Zero-shot fine-grained classification using Mistral')
    parser.add_argument('--data_path', type=str, 
                       default='pos_neg_cleaned.conll',
                       help='Path to the input CONLL file')
    parser.add_argument('--output_dir', type=str,
                       default='results_zero_finegrained',
                       help='Directory to save the results')
    # parser.add_argument('--limit', type=int, #to limit the number of examples for testing
    #                     default=None,
    #                     help='Limit number of examples for testing')
    return parser.parse_args()

def parse_conll_file(conll_file_path):
    data = []
    tokens = []
    ner_tags = []

    # Unified tag mapping (B- and I- merged)
    tag2id = {
        "O": 0,
        "Implicature": 1,
        "Ambiguity": 2,
        "Presupposition": 3,
        "Explicit": 4
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
                if tag in ["B-Implicature", "I-Implicature"]:
                    ner_tags.append(tag2id["Implicature"])
                elif tag in ["B-Ambiguity", "I-Ambiguity"]:
                    ner_tags.append(tag2id["Ambiguity"])
                elif tag in ["B-Presupposition", "I-Presupposition"]:
                    ner_tags.append(tag2id["Presupposition"])
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
id2label = {0: "O", 1: "Implicature", 2: "Ambiguity", 3: "Presupposition", 4: "Explicit"}
label2id = {"O": 0, "Implicature": 1, "Ambiguity": 2, "Presupposition": 3, "Explicit": 4}

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

def process_data(data):
    records = []
    for item in data:
        sentence = reconstruct_sentence(item["tokens"])
        label_seq = " ".join(id2label[t] for t in item["ner_tags"])
        records.append({"sentence": sentence, "ner_tag": label_seq})
    return pd.DataFrame(records)

def setup_model():
    model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = MistralTokenizer.from_hf_hub(model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config
    )
    return model, tokenizer

def build_prompt(sentence):
     prompt = f"""Your task is to analyze the following text and classify each sentence as one of the following categories: Implicature, Ambiguity, Presupposition, or Explicit.

Definitions:
- Implicature: universal knowledge, or information that can be inferred from the utterance and its context, though not explicitly stated.
- Ambiguity: text that has multiple possible interpretations or meanings, cannot be resolved from context alone, requiring further precision.
- Presupposition: background assumptions or information that is taken for granted, presenting a notion as already shared by the author and their addressee(s).
- Explicit: information that is clearly and directly stated.

Instructions:
- For each span found, return a separate JSON object with exactly two fields:
  - "sentence": the exact span from the input sentence expressing the content.
  - "type": "Implicature", "Ambiguity", "Presupposition", or "Explicit".
- If none of them is found, return one JSON object with both fields set to "".
- Do **not** wrap the JSON objects in a list (no square brackets).
- Separate multiple JSON objects with **commas and spaces only**, e.g.:
  {{ "sentence": "...", "type": "Implicature" }}, {{ "sentence": "...", "type": "Explicit" }}
- The output must be strictly valid JSON:
  - Use double quotes only
  - Close all braces correctly
  - Do not include trailing commas
- Do not include any explanation, notes, or extra text. Output **only** the JSON objects.

Sentence:
{sentence}
"""
     return prompt

def extract_mistral_sentences(output):
    output = output.strip()
    if output.lower().startswith("output:"):
        output = output[len("output:"):].strip()
    pattern = r'\{\s*"sentence"\s*:\s*"(.+?)"\s*,\s*"type"\s*:\s*"(.+?)"\s*\}'
    return [(s.strip(), t.strip().capitalize()) for s, t in re.findall(pattern, output, re.DOTALL)]

def get_sentence_level_labels(text, gold):
    tokens = word_tokenize(text.strip())
    labels = gold.strip().split()

    # Handle mismatch between tokens and labels
    if len(tokens) != len(labels):
        logging.warning(f"Token-label mismatch: {len(tokens)} tokens vs {len(labels)} labels")
        if len(tokens) > len(labels):
            # Pad labels with "O" for extra tokens
            labels.extend(["O"] * (len(tokens) - len(labels)))
        elif len(tokens) < len(labels):
            # Truncate labels to match the number of tokens
            labels = labels[:len(tokens)]

    sents = sent_tokenize(text.strip())
    output, idx = [], 0
    for sent in sents:
        n = len(word_tokenize(sent))
        tag_slice = labels[idx:idx+n]
        label = ("Implicature" if any("Implicature" in t for t in tag_slice)
                 else "Ambiguity" if any("Ambiguity" in t for t in tag_slice)
                 else "Presupposition" if any("Presupposition" in t for t in tag_slice)
                 else "Explicit" if any("Explicit" in t for t in tag_slice)
                 else "O")
        output.append((sent, label))
        idx += n
    return output

def best_match(pred_sent, gold_sents):
    best_idx, best_score = -1, 0
    for i, (gold_sent, _) in enumerate(gold_sents):
        score = SequenceMatcher(None, pred_sent.lower(), gold_sent.lower()).ratio()
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx if best_score > 0.8 else -1

def validate_file(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Input file not found: {file_path}")
        raise FileNotFoundError(f"Input file not found: {file_path}")

def save_predictions(predictions, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"finegrained_predictions_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=2)
    logging.info(f"Predictions saved to {output_file}")

def evaluate_predictions(predictions, output_dir):
    gold_all, pred_all = [], []
    for item in predictions:
        gold_pairs = get_sentence_level_labels(item["sentence"], item["gold"])
        pred_pairs = extract_mistral_sentences(item["mistral_output"])
        matched = set()
        pred_sent_map = {}
        for pred_sent, pred_label in pred_pairs:
            idx = best_match(pred_sent, gold_pairs)
            if idx != -1 and idx not in matched:
                pred_sent_map[idx] = pred_label
                matched.add(idx)
        for i, (gold_sent, gold_label) in enumerate(gold_pairs):
            pred_label = pred_sent_map.get(i, "O")
            gold_all.append(gold_label)
            pred_all.append(pred_label)

    report = classification_report(gold_all, pred_all, labels=["Implicature", "Ambiguity", "Presupposition", "Explicit", "O"], digits=4)
    logging.info("Evaluation completed")
    output_file = os.path.join(output_dir, "classification_report.txt")
    with open(output_file, "w") as f:
        f.write(report)
    logging.info(f"Classification report saved to {output_file}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    validate_file(args.data_path)
    data = parse_conll_file(args.data_path)
    df = process_data(data)

    # if args.limit:
    #     df = df.head(args.limit)
    #     logging.info(f"Limiting to first {args.limit} examples for testing.")

    # Load model and tokenizer
    model, tokenizer = setup_model()
    model.eval() 
    pad_token_id = None 

    predictions = []
    for row in tqdm(df.itertuples(), total=len(df)):
        prompt = build_prompt(row.sentence)

        messages = [{"role": "user", "content": prompt}]
        chat_request = ChatCompletionRequest(messages=messages)
        tokenized = tokenizer.encode_chat_completion(chat_request)
        input_ids = torch.tensor([tokenized.tokens], device=model.device) 

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=2048,
                do_sample=False
    )

        generated_tokens = outputs[0][len(tokenized.tokens):]
        clean = tokenizer.decode(generated_tokens).strip()

        predictions.append({
            "sentence": row.sentence,
            "gold": row.ner_tag,
            "mistral_output": clean
        })

    save_predictions(predictions, args.output_dir)
    evaluate_predictions(predictions, args.output_dir)

if __name__ == "__main__":
    main()