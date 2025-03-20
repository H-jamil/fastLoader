#!/bin/bash

# Activate pyenv environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate training_gpu

# Get the full path to Python after activation
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"

# Set PyTorch library path for any custom extensions
TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH

# Set up distributed training environment variables
# Get RANK and WORLD_SIZE from MPI environment
if [ -n "$OMPI_COMM_WORLD_RANK" ]; then
    export RANK=$OMPI_COMM_WORLD_RANK
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
elif [ -n "$PMI_RANK" ]; then
    export RANK=$PMI_RANK
    export WORLD_SIZE=$PMI_SIZE
else
    echo "Warning: No MPI environment detected"
fi

# Set master node information
export MASTER_ADDR="10.52.0.178" 
export MASTER_PORT="29500"

# Calculate LOCAL_RANK (assuming 1 GPU per process for simplicity)
if [ -n "$RANK" ]; then
    export LOCAL_RANK=$(($RANK % $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)))
    if [ -z "$LOCAL_RANK" ]; then
        export LOCAL_RANK=0  # Fallback if nvidia-smi fails
    fi
fi

# Print distributed environment for debugging
echo "Distributed environment:"
echo "RANK=$RANK, WORLD_SIZE=$WORLD_SIZE, LOCAL_RANK=$LOCAL_RANK"
echo "MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT"

# Execute the actual script with all arguments passed to this script
exec python "$@"
