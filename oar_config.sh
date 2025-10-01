NAME="mistral_zero_binary"
PROJECT_NAME="test"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/phd/test"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"
export HUGGINGFACE_HUB_TOKEN=$(cat /home/esvirido/.huggingface/token)

# Make sure the log directory exists
mkdir -p "$LOGDIR"


W_HOURS=3                 # Walltime in hours (increased for fine-tuning and evaluation)
L_NGPUS=1                  # Number of GPUs (increased for larger model)
P_MINCUDACAPABILITY=7      # Minimum compute capability (e.g., 7 for A100s or 1080Tis)
P_MINGPUMEMORY=20000       # Minimum GPU memory in MB (40 GB for larger model)

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
     echo 'Starting zero-shot...'; \
     python3 mistral_zero_binary.py \
        --data_path pos_neg_imp_exp.conll \
        --output_dir results_zero_binary \
     echo 'Zero-shot completed...'; \
    " \
)
    #--stdout=logs/%jobid%.stdout \
    #--stderr=logs/%jobid%.stderr \
   
# Print the job ID / submission output
echo "$OAR_OUT"

