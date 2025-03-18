#!/usr/bin/env python3
import os
import sys
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet50, ResNet50_Weights
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

def setup_distributed(rank, world_size):
    """Initialize PyTorch distributed process group"""
    # Set environment variables for better NCCL performance
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # FastLoader already initializes MPI, so we use MPI_COMM_WORLD
    if torch.cuda.is_available():
        # Use NCCL backend for GPU communications
        # Set a shorter timeout to detect issues faster (180 seconds)
        timeout = datetime.timedelta(seconds=180.0)
        dist.init_process_group("nccl", timeout=timeout)
    else:
        # Use Gloo backend for CPU communications
        dist.init_process_group("gloo")
    
    print(f"Initialized PyTorch distributed process group for rank {rank} of {world_size}")
    
    # Set seed for reproducibility
    torch.manual_seed(42 + rank)  # Different seed per rank but deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

def cleanup_distributed():
    """Clean up distributed resources"""
    if dist.is_initialized():
        dist.destroy_process_group()

def synchronize_model_params(model):
    """Make sure model parameters are synchronized across all processes"""
    for param in model.parameters():
        # Synchronize each parameter across processes
        dist.broadcast(param.data, 0)  # Broadcast from rank 0 to all others

def safe_barrier(timeout=10.0):
    """Perform a barrier with timeout to avoid hanging indefinitely"""
    try:
        # Try to sync processes with a timeout
        work = dist.barrier(async_op=True)
        success = work.wait(timeout=datetime.timedelta(seconds=timeout))
        if not success:
            print(f"Warning: Barrier timeout after {timeout} seconds, continuing anyway")
    except Exception as e:
        print(f"Warning: Barrier failed with {str(e)}, continuing anyway")

def main():
    """Main function to test FastLoader with distributed ResNet50 training"""
    parser = argparse.ArgumentParser(description='Train ResNet50 with FastLoader and Distributed Data Parallel')
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--socket_bs', type=int, default=131072,
                        help='NCCL socket buffer size')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--comm_timeout', type=float, default=60.0,
                        help='Communication timeout in seconds')
    args = parser.parse_args()
    
    # Set NCCL environment variables
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker,lo"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_BUFFER_SIZE"] = str(args.socket_bs)
    os.environ["NCCL_IB_TIMEOUT"] = "22"  # Shorter timeout for InfiniBand
    os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking wait
    
    # Record start time for entire run
    start_time = time.time()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Initialize the prefetcher
    prefetcher = Prefetcher()
    
    # Start timing for initialization
    init_start = time.time()
    
    # Initialize with dataset path and parameters
    prefetcher.initialize(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )
    
    # Get initialization time
    init_time = time.time() - init_start
    
    # Get MPI rank and world size
    rank = prefetcher.get_rank()
    world_size = prefetcher.get_world_size()
    
    # Initialize PyTorch distributed process group
    setup_distributed(rank, world_size)
    
    # Set up device for this process
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Get dataset info
    dataset_info = prefetcher.get_dataset_info()
    num_classes = dataset_info['num_classes']
    
    # Create model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Modify the final FC layer to match our number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Ensure all processes have the same initial model weights
    synchronize_model_params(model)
    
    # Wrap model with DDP with find_unused_parameters to prevent deadlocks
    model = DDP(model, 
                device_ids=[rank % torch.cuda.device_count()] if torch.cuda.is_available() else None,
                find_unused_parameters=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        print(f"=== Distributed ResNet50 Training with FastLoader ===")
        print(f"Dataset scan time: {init_time:.2f} seconds")
        print(f"Global number of samples: {dataset_info['num_samples']}")
        print(f"Number of classes: {num_classes}")
        print(f"Running with {world_size} MPI processes")
        print(f"Target image size: {target_size}")
        print(f"Device: {device}")
        print(f"Socket buffer size: {args.socket_bs}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Training metrics
    total_train_loss = 0
    total_train_correct = 0
    total_train_samples = 0
    
    # Function for checking if gradient got NaN or inf values
    def check_gradients():
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return False
        return True
    
    # Run training for specified epochs
    for ep in range(args.num_epochs):
        epoch_start = time.time()
        
        if rank == 0:
            print(f"\n=== Starting Epoch {ep+1}/{args.num_epochs} ===")
        
        # Set a common seed for this epoch based on epoch number to ensure all ranks get the same data order
        epoch_seed = 42 + ep  # Different seed per epoch but same across ranks
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        
        # Reset for new epoch
        prefetcher.reset_epoch()
        
        # Important: Make sure all processes are in sync before starting epoch
        # Use a safe barrier with timeout
        safe_barrier(timeout=10.0)
        
        # Start prefetching
        prefetcher.start_prefetching()
        
        # Set model to training mode
        model.train()
        
        # Process batches
        total_batches = 0
        total_images = 0
        epoch_loss = 0.0
        epoch_correct = 0
        
        # Keep track of communication errors
        comm_errors = 0
        
        # Limit max batches per epoch to a small, fixed number for better synchronization
        # This is a safety mechanism to prevent processes from getting too far out of sync
        max_batches_per_epoch = min(
            dataset_info['num_samples'] // (args.batch_size * world_size),
            145  # Limit to 145 batches which seems to work fine based on earlier logs
        )
        if dataset_info['num_samples'] % (args.batch_size * world_size) != 0:
            max_batches_per_epoch += 1
        
        # Ensure all ranks process the same exact number of batches
        # by broadcasting max_batches_per_epoch from rank 0
        max_batches_tensor = torch.tensor([max_batches_per_epoch], dtype=torch.long, device=device)
        dist.broadcast(max_batches_tensor, 0)
        max_batches_per_epoch = max_batches_tensor.item()
        
        if rank == 0:
            print(f"Processing exactly {max_batches_per_epoch} batches per process for this epoch")
        
        # Process only a fixed number of batches per epoch per rank
        while total_batches < max_batches_per_epoch:
            try:
                # Periodically synchronize to make sure processes are still in step
                if total_batches % 10 == 0:
                    safe_barrier(timeout=5.0)
                
                # Get next batch - now returns a list of tensors
                data_list, labels_list = prefetcher.get_next_batch()
                
                # Check for end of epoch
                if data_list is None:
                    # If we reach the end before getting enough batches, continue with dummy batches
                    # to ensure all processes do the same number of optimization steps
                    if total_batches < max_batches_per_epoch:
                        print(f"Rank {rank}: Reached end of data at batch {total_batches}/{max_batches_per_epoch}. Using dummy batches to complete epoch.")
                        # Create dummy batch of zeros
                        dummy_batch_size = args.batch_size
                        data_batch = torch.zeros((dummy_batch_size, 3, target_size[0], target_size[1]), device=device)
                        labels_batch = torch.zeros(dummy_batch_size, dtype=torch.long, device=device)
                    else:
                        break
                else:
                    # Preprocess the batch - resize and stack tensors
                    data_batch = preprocess_batch(data_list, target_size=target_size)
                    
                    # Convert labels list to tensor
                    labels_batch = torch.tensor(labels_list, dtype=torch.long)
                    
                    # Move tensors to the correct device
                    data_batch = data_batch.to(device)
                    labels_batch = labels_batch.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data_batch)
                loss = criterion(outputs, labels_batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Check for NaN or inf in gradients
                if not check_gradients():
                    print(f"Warning: Rank {rank} detected NaN or inf in gradients at batch {total_batches}. Skipping update.")
                    optimizer.zero_grad()  # Clear bad gradients
                else:
                    # Optimize
                    optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels_batch).sum().item()
                
                # Update counters
                total_batches += 1
                batch_size = len(data_list)
                total_images += batch_size
                epoch_loss += loss.item() * batch_size
                epoch_correct += batch_correct
                
                # Print progress
                if total_batches % 10 == 0 and rank == 0:
                    avg_loss = epoch_loss / total_images if total_images > 0 else 0
                    avg_acc = 100 * epoch_correct / total_images if total_images > 0 else 0
                    print(f"Rank 0: Batch {total_batches}: Processed {total_images} images, "
                          f"Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
                
                # Print first batch info (for debugging)
                if total_batches == 1 and rank == 0:
                    print_tensor_info(data_batch, "First image batch")
                    print_tensor_info(labels_batch, "First label batch")
                
            except Exception as e:
                print(f"Error processing batch at rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                comm_errors += 1
                
                # If too many errors, exit the loop
                if comm_errors > 5:
                    print(f"Rank {rank}: Too many errors ({comm_errors}), stopping epoch early")
                    break
                
                # Try to continue with the next batch
                continue
        
        # Stop prefetching
        prefetcher.stop_prefetching()
        
        # Wait for all processes to complete their epoch with a timeout
        safe_barrier(timeout=30.0)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Compute average loss and accuracy for this epoch
        avg_loss = epoch_loss / total_images if total_images > 0 else 0
        avg_acc = 100 * epoch_correct / total_images if total_images > 0 else 0
        
        # Update total metrics
        total_train_loss += epoch_loss
        total_train_correct += epoch_correct
        total_train_samples += total_images
        
        # Print performance metrics
        print(f"Node {rank} Epoch {ep+1} Performance:")
        print(f"  - Total batches: {total_batches}")
        print(f"  - Total images: {total_images}")
        print(f"  - Loss: {avg_loss:.4f}")
        print(f"  - Accuracy: {avg_acc:.2f}%")
        print(f"  - Epoch time: {epoch_time:.2f} seconds")
        
        if total_images > 0:
            print(f"  - Images/second: {total_images / epoch_time:.2f}")
        
        # Barrier to make sure all processes complete the epoch
        safe_barrier(timeout=30.0)
    
    # Record end time
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate overall training metrics
    overall_avg_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
    overall_avg_acc = 100 * total_train_correct / total_train_samples if total_train_samples > 0 else 0
    
    if rank == 0:
        print("\n=== Distributed Training Complete ===")
        print(f"Overall Training Loss: {overall_avg_loss:.4f}")
        print(f"Overall Training Accuracy: {overall_avg_acc:.2f}%")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
        print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")

    # Clean up
    cleanup_distributed()

if __name__ == "__main__":
    main()