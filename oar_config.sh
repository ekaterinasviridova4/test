NAME="mistral_eval_binary"
PROJECT_NAME="test"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/test"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"
export HUGGINGFACE_HUB_TOKEN=$(cat /home/esvirido/.huggingface/token)

# Make sure the log directory exists
mkdir -p "$LOGDIR"


W_HOURS=1                  # Walltime in hours
L_NGPUS=1                  # Number of GPUs (increased for larger model)
P_MINCUDACAPABILITY=7      # Minimum compute capability (e.g., 7 for A100s or 1080Tis)
P_MINGPUMEMORY=10000       # Minimum GPU memory in MB (40 GB for larger model)

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
     echo \"Hugging Face Token: \$HUGGINGFACE_HUB_TOKEN\"; \
     module load conda; \
     source /home/esvirido/miniconda3/bin/activate /home/esvirido/miniconda3/envs/llm-env; \
     python3 evaluate_mistral_finetuned_binary.py \
        --data_dir out_jsonl \
        --output_dir results_finetune_binary \
        --pred_dir results_finetune_binary/predictions \
        --split test" \
)
    #--stdout=logs/%jobid%.stdout \
    #--stderr=logs/%jobid%.stderr \
   
# Print the job ID / submission output
echo "$OAR_OUT"

