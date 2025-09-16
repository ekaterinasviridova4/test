from mistral_finetune_binary import parse_conll_file, prepare_and_save_splits

conll_path = "pos_neg_imp_exp.conll"
output_dir = "results_finetune_binary"

print(f"Preparing data from {conll_path}...")
data = parse_conll_file(conll_path)
prepare_and_save_splits(data, output_dir)
print(f"Done. Files saved to {output_dir}")