#!/usr/bin/env python3
import os
import re
import csv
import json
import argparse
from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ---------- IO ----------

def load_jsonl(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

# ---------- Span parsing & normalization ----------

TAG_PATTERN = re.compile(
    r"<\s*(Implicit|Explicit)\s*>\s*(.*?)\s*<\s*/\s*\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def normalize_space(s: str) -> str:
    # collapse whitespace, strip, and normalize quotes dashes lightly
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(s.split()).strip()

def tokenize(s: str) -> List[str]:
    s = normalize_space(s.lower())
    return TOKEN_PATTERN.findall(s)

def extract_spans_from_text(tagged_text: str) -> List[Dict]:
    """
    Returns list of dicts: {label: 'Implicit'|'Explicit', text: raw_span, tokens: [..]}
    Robust to extra whitespace/newlines and tag case.
    """
    spans = []
    for m in TAG_PATTERN.finditer(tagged_text):
        label = m.group(1).capitalize()  # normalize to 'Implicit'/'Explicit'
        span_text = m.group(2).strip()
        spans.append({
            "label": label,
            "text": span_text,
            "tokens": tokenize(span_text),
        })
    return spans

# ---------- Matching ----------

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def match_doc_spans(pred_spans: List[Dict], gold_spans: List[Dict], threshold: float = 0.8):
    """
    Greedy highest-overlap matching per document, label-must-match, thresholded.
    Returns:
      matches: list of tuples (gold_idx, pred_idx, gold_label, pred_label, overlap, gold_text, pred_text)
      unmatched_gold_idx: set of remaining gold indices
      unmatched_pred_idx: set of remaining pred indices
    """
    # Build candidate pairs only when labels match
    candidates = []
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            if g["label"] != p["label"]:
                continue
            overlap = jaccard(g["tokens"], p["tokens"])
            if overlap >= threshold:
                candidates.append((overlap, gi, pi))

    # Greedy: pick highest overlap pairs first
    candidates.sort(reverse=True)
    matched_g, matched_p = set(), set()
    matches = []
    for overlap, gi, pi in candidates:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi)
        matched_p.add(pi)
        g, p = gold_spans[gi], pred_spans[pi]
        matches.append((gi, pi, g["label"], p["label"], overlap, g["text"], p["text"]))

    unmatched_gold = set(range(len(gold_spans))) - matched_g
    unmatched_pred = set(range(len(pred_spans))) - matched_p
    return matches, unmatched_gold, unmatched_pred

# ---------- Evaluation pipeline ----------

def evaluate(pred_file: str, out_dir: str, threshold: float = 0.8):
    os.makedirs(out_dir, exist_ok=True)

    # Load predictions and gold data from the same file
    data_jsonl = load_jsonl(pred_file)

    y_true, y_pred = [], []
    exact_matches = 0
    partial_matches = 0

    # Diagnostics CSV
    diag_rows = [["doc_id", "type", "gold_label", "pred_label", "overlap", "gold_text", "pred_text"]]

    total_gold_spans = 0
    total_pred_spans = 0

    for i, entry in enumerate(data_jsonl):
        pred_text = entry.get("pred", "").strip()
        gold_text = entry.get("ref", "").strip()  # Use "ref" instead of "output"

        pred_spans = extract_spans_from_text(pred_text)
        gold_spans = extract_spans_from_text(gold_text)

        total_gold_spans += len(gold_spans)
        total_pred_spans += len(pred_spans)

        # Exact match accounting (normalized text + label, per document)
        gold_exact_pool = {(s["label"], normalize_space(s["text"]).lower()) for s in gold_spans}
        pred_exact_pool = {(s["label"], normalize_space(s["text"]).lower()) for s in pred_spans}
        exact_here = len(gold_exact_pool & pred_exact_pool)
        exact_matches += exact_here

        # Greedy label-constrained Jaccard matching
        matches, unmatched_gold, unmatched_pred = match_doc_spans(pred_spans, gold_spans, threshold=threshold)
        partial_matches += len(matches)

        # Build classification vectors:
        # - Matched spans contribute gold_label vs pred_label (they’re same by design).
        for gi, pi, gl, pl, ov, gt, pt in matches:
            y_true.append(gl)
            y_pred.append(pl)
            diag_rows.append([i, "MATCH", gl, pl, f"{ov:.3f}", gt, pt])

        # - Unmatched gold spans count as FN: true label vs predicted 'O'
        for gi in unmatched_gold:
            gl = gold_spans[gi]["label"]
            gt = gold_spans[gi]["text"]
            y_true.append(gl)
            y_pred.append("O")
            diag_rows.append([i, "MISS", gl, "O", "0.000", gt, ""])

        # - Unmatched pred spans count as FP: true 'O' vs predicted label
        for pi in unmatched_pred:
            pl = pred_spans[pi]["label"]
            pt = pred_spans[pi]["text"]
            y_true.append("O")
            y_pred.append(pl)
            diag_rows.append([i, "SPURIOUS", "O", pl, "0.000", "", pt])

    # Reports
    labels_all = ["Implicit", "Explicit", "O"]
    print("\n=== Span-level report (partial credit with Jaccard ≥ {:.2f}) ===".format(threshold))
    print(classification_report(y_true, y_pred, labels=labels_all, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("Labels:", labels_all)
    print(cm)

    # Save reports
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, labels=labels_all, digits=3, zero_division=0))
        f.write("\n\nLabels: " + ", ".join(labels_all) + "\n")
        f.write(np.array2string(cm, separator=", "))

    with open(os.path.join(out_dir, "confusion_matrix.txt"), "w") as f:
        f.write("Labels: " + ", ".join(labels_all) + "\n")
        f.write(np.array2string(cm, separator=", "))

    # Exact vs partial summary
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Documents compared: {len(data_jsonl)}\n")
        f.write(f"Total gold spans: {total_gold_spans}\n")
        f.write(f"Total pred spans: {total_pred_spans}\n")
        f.write(f"Exact matches (label + normalized text): {exact_matches}\n")
        f.write(f"Partial matches (label + Jaccard≥{threshold}): {partial_matches}\n")

    # Diagnostics CSV for manual inspection
    with open(os.path.join(out_dir, "diagnostics.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(diag_rows)

# ---------- CLI ----------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Span-level evaluation for <Implicit>/<Explicit> tagging.")
    p.add_argument("--predictions_file", required=True, help="JSONL with lines like {'pred': '<tagged text>', 'ref': '<tagged text>'}")
    p.add_argument("--output_dir", required=True, help="Where to save reports")
    p.add_argument("--threshold", type=float, default=0.8, help="Jaccard token-overlap threshold")
    args = p.parse_args()

    evaluate(args.predictions_file, args.output_dir, threshold=args.threshold)

# import os
# import json
# import re
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix

# def load_jsonl_data(file_path):
#     """Load data from a JSONL file."""
#     data = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# def extract_spans(data):
#     """Extract spans and their labels from the data."""
#     spans = []
#     implicit_pattern = re.compile(r"<Implicit>\s*(.*?)\s*</Implicit>", re.DOTALL)
#     explicit_pattern = re.compile(r"<Explicit>\s*(.*?)\s*</Explicit>", re.DOTALL)

#     for text in data:
#         implicit_matches = implicit_pattern.findall(text)
#         explicit_matches = explicit_pattern.findall(text)

#         spans.extend([("Implicit", span.strip().split()) for span in implicit_matches])
#         spans.extend([("Explicit", span.strip().split()) for span in explicit_matches])

#     return spans

# def token_overlap(span1, span2):
#     """Calculate token-level overlap between two spans."""
#     set1, set2 = set(span1), set(span2)
#     return len(set1 & set2) / len(set1 | set2)

# def evaluate_spans(pred_spans, ref_spans, threshold=0.8):
#     """Evaluate predicted spans against reference spans."""
#     y_true, y_pred = [], []

#     matched_preds = set()
#     for ref_label, ref_tokens in ref_spans:
#         matched = False
#         for i, (pred_label, pred_tokens) in enumerate(pred_spans):
#             if i in matched_preds:
#                 continue
#             overlap = token_overlap(ref_tokens, pred_tokens)
#             if overlap >= threshold and ref_label == pred_label:
#                 y_true.append(ref_label)
#                 y_pred.append(pred_label)
#                 matched_preds.add(i)
#                 matched = True
#                 break
#         if not matched:
#             y_true.append(ref_label)
#             y_pred.append("O")  # No matching prediction, set to 'O'

#     # Penalize unmatched predictions
#     for i, (pred_label, pred_tokens) in enumerate(pred_spans):
#         if i not in matched_preds:
#             y_true.append("O")
#             y_pred.append(pred_label)

#     return y_true, y_pred

# def main(predictions_file, gold_file, output_dir, threshold=0.8):
#     os.makedirs(output_dir, exist_ok=True)

#     # Load predictions and gold data
#     predictions = [entry["pred"] for entry in load_jsonl_data(predictions_file)]
#     gold = [entry["output"] for entry in load_jsonl_data(gold_file)]

#     # Extract spans from predictions and references
#     pred_spans = extract_spans(predictions)
#     ref_spans = extract_spans(gold)

#     # Evaluate spans
#     y_true, y_pred = evaluate_spans(pred_spans, ref_spans, threshold=threshold)

#     # Generate fine-grained classification report
#     fine_grained_report = classification_report(y_true, y_pred, digits=3)
#     print(fine_grained_report)

#     # Save the fine-grained report to a file
#     report_path = os.path.join(output_dir, "classification_report.txt")
#     with open(report_path, "w") as f:
#         f.write(fine_grained_report)

#     # --- Confusion Matrix ---
#     labels = ["Implicit", "Explicit", "O"]
#     cm = confusion_matrix(y_true, y_pred, labels=labels)

#     cm_report = "Confusion Matrix (rows = true, cols = predicted):\n"
#     cm_report += "Labels: " + ", ".join(labels) + "\n"
#     cm_report += np.array2string(cm, separator=", ")

#     print(cm_report)

#     # Save confusion matrix
#     cm_path = os.path.join(output_dir, "confusion_matrix.txt")
#     with open(cm_path, "w") as f:
#         f.write(cm_report)

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Evaluate predictions against gold data.")
#     parser.add_argument("--predictions_file", type=str, required=True, help="Path to the predictions JSONL file.")
#     parser.add_argument("--gold_file", type=str, required=True, help="Path to the gold JSONL file.")
#     parser.add_argument("--output_dir", type=str, required=True, help="Directory to save evaluation results.")
#     parser.add_argument("--threshold", type=float, default=0.8, help="Token overlap threshold for matching spans.")

#     args = parser.parse_args()
#     main(
#         predictions_file=args.predictions_file,
#         gold_file=args.gold_file,
#         output_dir=args.output_dir,
#         threshold=args.threshold,
#     )
