import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from distributed_loader import DistributedDataLoader
import argparse
import time

def setup(rank, world_size):
    # Remove redundant initialization since it's done in main()
    pass

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, dataset_path):
    # Setup GPU device
    device = setup(rank, world_size)
    
    # Create data loader with enhanced prefetching
    batch_size = 32
    loader = DistributedDataLoader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=4,  # Increased for better performance
        prefetch_factor=4  # Increased for better performance
    )
    
    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        # Set epoch for proper shuffling
        loader.set_epoch(epoch)
        
        print(f"Node {rank}, Starting epoch {epoch+1}")
        
        try:
            for batch_idx, (images, labels) in enumerate(loader):
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if batch_idx % 20 == 0:
                    current_time = time.time()
                    images_per_second = (batch_idx + 1) * batch_size / (current_time - loader.epoch_start_time)
                    print(f"Node {rank}, Epoch {epoch+1}, "
                          f"Batch {batch_idx}, "
                          f"Batch size: {images.size(0)}, "
                          f"Images/sec: {images_per_second:.2f}")
                
                # Your training code would go here
                # ...
                
                # Update epoch statistics
                loader.update_epoch_stats(images.size(0))
                
        except Exception as e:
            print(f"Error in training loop at rank {rank}, epoch {epoch+1}: {str(e)}")
            # Try to recover and continue
            continue
            
        # Get and print epoch statistics
        epoch_stats = loader.get_epoch_stats()
        if epoch_stats:
            print(f"\nNode {rank}, Epoch {epoch+1} Summary:")
            print(f"  Duration: {epoch_stats['epoch_duration']:.2f} seconds")
            print(f"  Total Images: {epoch_stats['total_images']}")
            print(f"  Average Speed: {epoch_stats['images_per_second']:.2f} images/second")
            
        # Synchronize at epoch end with proper device
        dist.barrier(device_ids=[rank])
        
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-rank', type=int)
    parser.add_argument('--master-addr', type=str)
    parser.add_argument('--master-port', type=str)
    parser.add_argument('--world-size', type=int)
    parser.add_argument('--dataset-path', type=str)
    args = parser.parse_args()    
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # Initialize process group
    dist.init_process_group(
        "nccl",
        rank=args.node_rank,
        world_size=args.world_size
    )
    
    # Create data loader and start training
    train(args.node_rank, args.world_size, args.dataset_path)

if __name__ == "__main__":
    main()
