#!/bin/bash

# Get PyTorch library path
TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
echo "PyTorch library path: $TORCH_LIB_PATH"

# Set LD_LIBRARY_PATH to include PyTorch libraries
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH

# Run the Python script
exec python "$@" 