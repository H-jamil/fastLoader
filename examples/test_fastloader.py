#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from fastloader import Prefetcher

def print_tensor_info(tensor, name):
    """Print information about a PyTorch tensor"""
    if tensor is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")

def preprocess_batch(tensor_list, target_size=(224, 224)):
    """Resize a list of tensors to the same size and stack them"""
    if not tensor_list:
        return None
    
    # Resize all tensors to the target size
    resized_tensors = []
    for tensor in tensor_list:
        # Use F.interpolate to resize the image tensor
        # Input needs to be [batch_size, channels, height, width], but our tensor is [channels, height, width]
        # So we need to add a batch dimension first
        resized = F.interpolate(
            tensor.unsqueeze(0),  # Add batch dimension
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        resized_tensors.append(resized)
    
    # Stack resized tensors into a batch
    return torch.stack(resized_tensors)

def main():
    """Main function to test FastLoader"""
    parser = argparse.ArgumentParser(description='Test FastLoader with CephFS')
    parser.add_argument('--data_path', type=str, default='/mnt/cephfs/subset/train/',
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prefetching')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker threads')
    parser.add_argument('--prefetch_factor', type=int, default=8,
                        help='Number of batches to prefetch')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to run')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='Target size for resizing images, format: width,height')
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Initialize the prefetcher
    prefetcher = Prefetcher()
    
    # Start timing
    start_time = time.time()
    
    # Initialize with dataset path and parameters
    prefetcher.initialize(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )
    
    # Get initialization time
    init_time = time.time() - start_time
    
    # Get MPI rank and world size
    rank = prefetcher.get_rank()
    world_size = prefetcher.get_world_size()
    
    # Get dataset info
    dataset_info = prefetcher.get_dataset_info()
    
    if rank == 0:
        print(f"=== Testing FastLoader Python Interface with CephFS ===")
        print(f"Dataset scan time: {init_time:.2f} seconds")
        print(f"Global number of samples: {dataset_info['num_samples']}")
        print(f"Number of classes: {dataset_info['num_classes']}")
        print(f"Running with {world_size} MPI processes")
        print(f"Target image size: {target_size}")
    
    # Run multiple epochs
    for ep in range(args.num_epochs):
        if rank == 0:
            print(f"\n=== Starting Epoch {ep+1} ===")
        
        # Reset for new epoch
        prefetcher.reset_epoch()
        
        # Start prefetching
        prefetcher.start_prefetching()
        
        # Process batches
        epoch_start = time.time()
        total_batches = 0
        total_images = 0
        
        # Adjust workers based on epoch (example)
        if ep == 2 and rank == 0:
            print(f"Increasing number of workers to 12")
            prefetcher.set_num_workers(12)
        
        while True:
            try:
                # Get next batch - now returns a list of tensors
                data_list, labels_list = prefetcher.get_next_batch()
                
                # Check for end of epoch
                if data_list is None:
                    break
                
                # Preprocess the batch - resize and stack tensors
                data_batch = preprocess_batch(data_list, target_size=target_size)
                
                # Convert labels list to tensor
                labels_batch = torch.tensor(labels_list)
                
                total_batches += 1
                total_images += len(data_list)
                
                # Print progress
                if total_batches % 10 == 0 and rank == 0:
                    print(f"Processed {total_batches} batches, {total_images} images")
                
                # Print first batch info (for debugging)
                if total_batches == 1 and rank == 0:
                    print_tensor_info(data_batch, "First image batch")
                    print_tensor_info(labels_batch, "First label batch")
                
            except Exception as e:
                print(f"Error processing batch at rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Stop prefetching
        prefetcher.stop_prefetching()
        
        # Print performance metrics
        print(f"Node {rank} Performance:")
        print(f"  - Total batches: {total_batches}")
        print(f"  - Total images: {total_images}")
        print(f"  - Epoch time: {epoch_time:.2f} seconds")
        
        if total_images > 0:
            print(f"  - Images/second: {total_images / epoch_time:.2f}")
    
    if rank == 0:
        print("\n=== FastLoader Test Complete ===")
    end_time = time.time()

    print(f"-----start time {start_time} and end time {end_time} total duration {end_time-start_time}")
if __name__ == "__main__":
    main() 
