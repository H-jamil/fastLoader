#!/bin/bash
# Script to install FastLoader on multiple machines

# Configuration
MACHINES=("10.52.0.178" "10.52.2.75")
REMOTE_USER="cc"  # Change to your remote username
REPO_DIR="~/fastLoader"

# Function to check if a command succeeded
check_command() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Build locally first to ensure it works
echo "Building locally..."
pip install -e .
check_command "Local build failed"

# Install on each remote machine
for machine in "${MACHINES[@]}"; do
    echo "Installing on $machine..."
    
    # Create directory if it doesn't exist
    ssh ${REMOTE_USER}@${machine} "mkdir -p ${REPO_DIR}"
    check_command "Failed to create directory on $machine"
    
    # Copy files to remote machine
    rsync -avz --exclude 'build' --exclude '*.egg-info' \
          --exclude '__pycache__' --exclude '.git' \
          ./ ${REMOTE_USER}@${machine}:${REPO_DIR}/
    check_command "Failed to copy files to $machine"
    
    # Install on remote machine
    ssh ${REMOTE_USER}@${machine} "cd ${REPO_DIR} && pip install -e ."
    check_command "Failed to install on $machine"
    
    echo "Installation on $machine completed successfully."
done

echo "All installations completed."

# Instructions for running distributed test
echo "
To run distributed test across machines:
1. Create a hostfile with the list of machines
   (e.g. echo '10.52.0.178:2' > hostfile)
2. Run with mpirun:
   mpirun -hostfile hostfile python examples/test_fastloader.py --data_path=/path/to/dataset
" 