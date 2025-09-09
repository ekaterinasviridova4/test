NAME="mistral_zero_binary"
PROJECT_NAME="test"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/test"
EMAIL="ekaterina.sviridova@inria.fr"
LOGDIR="$HOME/logs"

# Make sure the log directory exists
mkdir -p "$LOGDIR"

### Resource settings
W_HOURS=5                  # Walltime in hours
L_NGPUS=2                  # Number of GPUs
P_MINCUDACAPABILITY=7      # Minimum compute capability (e.g., 7 for A100s or 1080Tis)
P_MINGPUMEMORY=11000       # Minimum GPU memory in MB (11 GB)

# Submit the job
OAR_OUT=$(oarsub \
    --name "$NAME" \
    --directory "$PROJECT_DIR" \
    --stdout="$LOGDIR/%jobid%.stdout" \
    --stderr="$LOGDIR/%jobid%.stderr" \
    --property="gpu_compute_capability>='$P_MINCUDACAPABILITY' and gpu_mem>='$P_MINGPUMEMORY'" \
    --l "nodes=1/gpu=$L_NGPUS,walltime=$W_HOURS" \
    --notify "[ERROR,INFO]mail:$EMAIL" \
    "module load conda; conda activate llm_env; python3 mistral_zero_binary.py --limit 10" \
)
    #--stdout=logs/%jobid%.stdout \
    #--stderr=logs/%jobid%.stderr \
   
# Print the job ID / submission output
echo "$OAR_OUT"

