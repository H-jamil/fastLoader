#!/usr/bin/env python3
"""
Compare the performance of C++ and Python preprocessing.
This script demonstrates the preprocessing capability in the C++ implementation
versus doing it in Python.
"""
import os
import time
import argparse
import torch
import torch.nn.functional as F
from fastloader import Prefetcher

def preprocess_batch_python(tensor_list, target_size=(224, 224)):
    """Resize a list of tensors to the same size in Python"""
    if not tensor_list:
        return None
    
    # Resize all tensors to the target size
    resized_tensors = []
    for tensor in tensor_list:
        # Use F.interpolate to resize the image tensor
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
    """Main function to compare preprocessing performance"""
    parser = argparse.ArgumentParser(description='Compare C++ and Python preprocessing performance')
    parser.add_argument('--data_path', type=str, default='/mnt/cephfs/subset/train/',
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prefetching')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--num_batches', type=int, default=10,
                        help='Number of batches to process')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='Target size for resizing images, format: width,height')
    parser.add_argument('--print_shapes', action='store_true',
                        help='Print tensor shapes for verification')
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Initialize prefetcher for Python preprocessing
    print("\n=== Running with Python preprocessing ===")
    prefetcher_py = Prefetcher()
    prefetcher_py.initialize(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=False  # Disable C++ preprocessing
    )
    
    # Start prefetching
    prefetcher_py.start_prefetching()
    
    # Process batches with Python preprocessing
    py_start_time = time.time()
    python_total_time = 0
    
    for i in range(args.num_batches):
        # Get raw tensors
        data_list, labels_list = prefetcher_py.get_next_batch()
        
        if data_list is None:
            print("End of dataset reached before processing requested batches")
            break
        
        # Measure Python preprocessing time
        py_preprocess_start = time.time()
        data_batch = preprocess_batch_python(data_list, target_size)
        python_total_time += time.time() - py_preprocess_start
        
        if args.print_shapes and i == 0:
            print(f"Raw tensor example shape: {data_list[0].shape}")
            print(f"Preprocessed batch shape: {data_batch.shape}")
    
    # Stop prefetching
    prefetcher_py.stop_prefetching()
    
    py_total_time = time.time() - py_start_time
    py_avg_time = python_total_time / args.num_batches
    
    print(f"Python preprocessing:")
    print(f"  - Total time: {py_total_time:.4f} seconds")
    print(f"  - Preprocessing time: {python_total_time:.4f} seconds")
    print(f"  - Average preprocessing time per batch: {py_avg_time:.4f} seconds")
    
    # Now initialize prefetcher with C++ preprocessing
    print("\n=== Running with C++ preprocessing ===")
    prefetcher_cpp = Prefetcher()
    prefetcher_cpp.initialize(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        preprocess=True,  # Enable C++ preprocessing
        preprocess_size=target_size
    )
    
    # Reset for fair comparison
    prefetcher_cpp.reset_epoch()
    
    # Start prefetching
    prefetcher_cpp.start_prefetching()
    
    # Process batches with C++ preprocessing
    cpp_start_time = time.time()
    
    for i in range(args.num_batches):
        # Get preprocessed tensors
        data_list, labels_list = prefetcher_cpp.get_next_batch()
        
        if data_list is None:
            print("End of dataset reached before processing requested batches")
            break
        
        # Just stack the already-preprocessed tensors
        data_batch = torch.stack(data_list)
        
        if args.print_shapes and i == 0:
            print(f"Preprocessed tensor example shape: {data_list[0].shape}")
            print(f"Stacked batch shape: {data_batch.shape}")
    
    # Stop prefetching
    prefetcher_cpp.stop_prefetching()
    
    cpp_total_time = time.time() - cpp_start_time
    
    print(f"C++ preprocessing:")
    print(f"  - Total time: {cpp_total_time:.4f} seconds")
    
    # Compare the performance
    print("\n=== Performance Comparison ===")
    speedup = py_total_time / cpp_total_time if cpp_total_time > 0 else float('inf')
    print(f"Speed-up with C++ preprocessing: {speedup:.2f}x")
    print(f"C++ preprocessing is {speedup:.2f} times faster than Python preprocessing")

if __name__ == "__main__":
    main() 