NAME="mistral_7b_micro_finegrained_binary"
PROJECT_NAME="test"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/phd/test"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"
export HUGGINGFACE_HUB_TOKEN=$(cat /home/esvirido/.huggingface/token)

# Make sure the log directory exists
mkdir -p "$LOGDIR"

# Mistral 7B specific directories
MODEL_NAME="mistral-7b"
OUTPUT_DIR="7B_Mistral_Llama/results_micro_mistral-7b_finetune_binary"
DATA_DIR="out_combined_binary_jsonl"

W_HOURS=10                 # Walltime in hours
L_NGPUS=1                  # Number of GPUs (1 is sufficient with LoRA + quantization)
P_MINCUDACAPABILITY=7      # Minimum compute capability
P_MINGPUMEMORY=24000       # Minimum GPU memory in MB (24 GB should be enough)

# Submit the job
OAR_OUT=$(oarsub \
    --name "$NAME" \
    --directory "$PROJECT_DIR" \
    --stdout="$LOGDIR/%jobid%.stdout" \
    --stderr="$LOGDIR/%jobid%.stderr" \
    --property="gpu_compute_capability>='$P_MINCUDACAPABILITY' and gpu_mem>='$P_MINGPUMEMORY'" \
    --l "nodes=1/gpu=$L_NGPUS,walltime=$W_HOURS" \
    --notify "[ERROR,INFO]mail:$EMAIL" \
    "export HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN; \
     module load conda; \
     source /home/esvirido/miniconda3/bin/activate /home/esvirido/miniconda3/envs/llm-env; \
     
     echo 'Starting Mistral 7B fine-tuning...'; \
     python3 7B_Mistral_Llama/finetune_binary.py \
        --model_name $MODEL_NAME \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR; \
     
     echo 'Starting Mistral 7B evaluation...'; \
     python3 7B_Mistral_Llama/evaluate_finetuned_binary.py \
        --model_name $MODEL_NAME \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --pred_dir $OUTPUT_DIR/predictions \
        --split test; \
     
     echo 'Mistral 7B completed successfully!'
    " \
)

# Print the job ID / submission output
echo "$OAR_OUT"