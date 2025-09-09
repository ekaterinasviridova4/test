NAME="mistral_zero_binary"
PROJECT_NAME="test"
HOME="/home/esvirido"
PROJECT_DIR="$HOME/test"
EMAIL="ekaterina.sviridova@inria.fr"

### Properties
# Mind that asking for more time than you need may make your job wait longer in
# the queue. It doesn't affect your karma (only the resources that you actually
# use do that).
W_HOURS=5
# For multi-GPU jobs, some stuff has to be adapted. E.g., you need to specify
# that you want the GPUs to be on the same node. Check the FAQ for more details.
L_NGPUS=2
# The following specifies "how good of a GPU". See the table named "Compute
# Capability, GPU semiconductors and Nvidia GPU board products" at
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported 6.1 requires at least a GTX
# 1080 Ti. There are many of these in the cluster: See what's available in at
# https://wiki.inria.fr/ClustersSophia/Hardware
P_MINCUDACAPABILITY=7
# Minimum GPU memory (in MB)``
# 11 GB was also chosen to include the GTX 1080 Ti (it has 11 GB = 11264 MB)
P_MINGPUMEMORY=11000

# Now we create a variable containing a huge string containing the command to be
# executed in the cluster It consists of the ⁠ oarsub ⁠ command followed by a
# bunch of options using the variables we defined above Notice the
# ⁠ --array-param-file ⁠ option. More on that at the end. The ⁠ $1 ⁠ in shell
# scripts is substituted by the first argument passed to the script (it's like
# ⁠ sys.argv[1] ⁠ in Python). The idea is that we can use this script as
# ⁠ ./test_on_cluster.sh SOMETHING.sh ⁠ and it will run ⁠ SOMETHING.sh ⁠ in the
# cluster.
# when you want to use only 1 gpu, substitute this command: -l gpu=1,walltime=$W_HOURS \
# -l gpu=1,walltime=$W_HOURS \ use the quotes
# when not using mistral small: "module load conda; conda activate rag_counterspeech; python3 summarize_para_mistral.py \ with the quotes
OAR_OUT=$(oarsub \
    --name "$NAME" \
    --directory "$PROJECT_DIR" \
    --stdout=logs/%jobid%.stdout \
    --stderr=logs/%jobid%.stderr \
    --property="gpu_compute_capability>='$P_MINCUDACAPABILITY' and gpu_mem>='$P_MINGPUMEMORY'" \
    --l "nodes=1/gpu=$L_NGPUS,walltime=$W_HOURS" \
    --notify "[ERROR,INFO]mail:$EMAIL" \
    "module load conda; conda activate llm_env; python3 mistral_zero_binary.py" \
)

# Run the string contained in ⁠ OAR_OUT ⁠ as a command. I don't remember why I did
# like this (defining a string and then running it) but it was necessary for
# something specific case. In particular, the FAQ does things differently.

#module load cuda/12.2
#module load conda
echo "$OAR_OUT"

# This is supposed to be used as ⁠ ./submit_array_to_cluster.sh ./S.sh ⁠ for some
# script ⁠ S.sh ⁠ that you want to run in the cluster. It will actually submit
# multiple jobs, one for each line in ⁠ args.txt ⁠. The ⁠ --array-param-file ⁠
# option is responsible for that. For example, if ⁠ args.txt ⁠ contains the
# following:
#
# arg2=4 argstring="hello"
