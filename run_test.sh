#!/bin/bash

TORCH_LIB_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))")
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH
python examples/test_fastloader.py "$@"
