#!/bin/bash

# Function to check if a port is available
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 1
    fi
    return 0
}

# Function to find an available port
find_available_port() {
    local start_port=12355
    local max_port=65535
    local port=$start_port
    
    while [ $port -le $max_port ]; do
        if check_port $port; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
    return 1
}

# Function to validate IP address
validate_ip() {
    local ip=$1
    if [[ $ip =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        return 0
    fi
    return 1
}

# Get master IP from environment or use default
MASTER_ADDR=${MASTER_ADDR:-"10.52.0.178"}
if ! validate_ip $MASTER_ADDR; then
    echo "Error: Invalid MASTER_ADDR IP address: $MASTER_ADDR"
    exit 1
fi

# Find available port
MASTER_PORT=$(find_available_port)
if [ $? -ne 0 ]; then
    echo "Error: Could not find an available port"
    exit 1
fi

# Node 1 (Master)
if [ "$1" == "0" ]; then
    NODE_RANK=0
    WORLD_SIZE=2
    
    # Start master process
    echo "Starting master node on $MASTER_ADDR:$MASTER_PORT"
    python train.py \
        --node-rank $NODE_RANK \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT \
        --world-size $WORLD_SIZE \
        --dataset-path /mnt/cephfs/subset/train/ || {
        echo "Error: Master node failed to start"
        exit 1
    }

# Node 2
elif [ "$1" == "1" ]; then
    NODE_RANK=1
    WORLD_SIZE=2
    
    # Wait for master to be ready
    echo "Waiting for master node to be ready..."
    while ! nc -z $MASTER_ADDR $MASTER_PORT; do
        sleep 1
    done
    
    # Start worker process
    echo "Starting worker node"
    python train.py \
        --node-rank $NODE_RANK \
        --master-addr $MASTER_ADDR \
        --master-port $MASTER_PORT \
        --world-size $WORLD_SIZE \
        --dataset-path /mnt/cephfs/subset/train/ || {
        echo "Error: Worker node failed to start"
        exit 1
    }
else
    echo "Error: Invalid node rank. Use 0 for master or 1 for worker"
    exit 1
fi
