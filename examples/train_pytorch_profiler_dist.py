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
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import glob
from PIL import Image

# Additional imports for profiling
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import tensorboard_trace_handler, schedule


def print_tensor_info(tensor, name):
    """Print information about a PyTorch tensor"""
    if tensor is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")


class ImageNetSubsetDataset(Dataset):
    """Custom dataset similar to ImageFolder but with more control over data loading"""
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Get all class directories (assuming ImageNet-like structure)
        self.class_dirs = sorted([d for d in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(d)])
        self.class_to_idx = {os.path.basename(d): i for i, d in enumerate(self.class_dirs)}
        
        # Get all image files
        self.samples = []
        for class_dir in self.class_dirs:
            class_idx = self.class_to_idx[os.path.basename(class_dir)]
            image_paths = glob.glob(os.path.join(class_dir, "*.JPEG"))
            image_paths.extend(glob.glob(os.path.join(class_dir, "*.jpeg")))
            image_paths.extend(glob.glob(os.path.join(class_dir, "*.jpg")))
            image_paths.extend(glob.glob(os.path.join(class_dir, "*.png")))
            for img_path in image_paths:
                self.samples.append((img_path, class_idx))
        
        print(f"Found {len(self.samples)} samples across {len(self.class_dirs)} classes in {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Apply transformations if any
            if self.transform:
                img = self.transform(img)
            
            return img, class_idx
        except Exception as e:
            # If an image fails to load, return a blank tensor and the class index
            print(f"Error loading image {img_path}: {e}")
            if self.transform:
                # Create a blank image with the right dimensions
                blank_img = torch.zeros(3, 224, 224)
                return blank_img, class_idx
            else:
                # Create a default blank PIL image
                blank_img = Image.new('RGB', (224, 224), color='black')
                return blank_img, class_idx


def setup_distributed(rank, world_size):
    """Initialize PyTorch distributed process group"""
    # Set environment variables for better NCCL performance
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    
    # Initialize the distributed process group
    if torch.cuda.is_available():
        # Use NCCL backend for GPU communications
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
    """Main function to test PyTorch distributed ResNet50 training with profiling"""
    parser = argparse.ArgumentParser(description='Train ResNet50 with PyTorch Distributed Data Parallel and Profiling')
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
    # Profiler arguments
    parser.add_argument('--profile', action='store_true',
                        help='Enable PyTorch profiling')
    parser.add_argument('--profile_folder', type=str, default='./pytorch_profiler_logs',
                        help='Folder to save profiler results')
    parser.add_argument('--profile_epochs', type=int, default=1,
                        help='Number of epochs to profile')
    parser.add_argument('--profile_batches', type=int, default=50,
                        help='Number of batches to profile')
    args = parser.parse_args()
    
    # Set NCCL environment variables
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker,lo"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_BUFFER_SIZE"] = str(args.socket_bs)
    os.environ["NCCL_IB_TIMEOUT"] = "22"  # Shorter timeout for InfiniBand
    os.environ["NCCL_BLOCKING_WAIT"] = "0"  # Non-blocking wait
    
    # Setup for distributed training
    # Get local rank from environment variable
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Record start time for entire run
    start_time = time.time()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Start timing for initialization
    init_start = time.time()
    
    # Load dataset
    dataset = ImageNetSubsetDataset(args.data_path, transform=transform)
    
    # Get initialization time
    init_time = time.time() - init_start
    
    # Initialize distributed process group
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # If not running with mpirun, default to single process
        rank = 0
        world_size = 1
    
    # Initialize PyTorch distributed process group
    setup_distributed(rank, world_size)
    
    # Create a DistributedSampler to handle data partitioning across processes
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False
    )
    
    # Create DataLoader with the distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False
    )
    
    # Set up device for this process
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Collect dataset info
    dataset_info = {
        'num_samples': len(dataset),
        'num_classes': len(dataset.class_dirs)
    }
    
    # Create model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Modify the final FC layer to match our number of classes
    model.fc = nn.Linear(model.fc.in_features, dataset_info['num_classes'])
    
    # Move model to device
    model = model.to(device)
    
    # Ensure all processes have the same initial model weights
    synchronize_model_params(model)
    
    # Wrap model with DDP
    model = DDP(model, 
                device_ids=[rank % torch.cuda.device_count()] if torch.cuda.is_available() else None,
                find_unused_parameters=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        print(f"=== Distributed ResNet50 Training with PyTorch + Profiling ===")
        print(f"Dataset scan time: {init_time:.2f} seconds")
        print(f"Global number of samples: {dataset_info['num_samples']}")
        print(f"Number of classes: {dataset_info['num_classes']}")
        print(f"Running with {world_size} processes")
        print(f"Target image size: {target_size}")
        print(f"Device: {device}")
        if args.profile:
            print(f"Profiling enabled. Results will be saved to {args.profile_folder}")
    
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
    
    # Create profiler if enabled
    profiler = None
    if args.profile:  # Only profile rank 0 for simplicity
        # Create directory for profiler results
        os.makedirs(args.profile_folder, exist_ok=True)
        
        # Define profiler schedule
        prof_schedule = schedule(
            wait=10,  # Skip first 10 batches to avoid warmup overhead
            warmup=5,  # Warmup for 5 batches
            active=args.profile_batches,  # Profile specified number of batches
            repeat=1,  # Profile one iteration
            skip_first=10  # Skip the first iterations for more stable profiling
        )
        
        # Create the profiler
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(args.profile_folder),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        )
    
    # Run training for specified epochs
    for ep in range(args.num_epochs):
        epoch_start = time.time()
        
        if rank == 0:
            print(f"\n=== Starting Epoch {ep+1}/{args.num_epochs} ===")
        
        # Set the epoch for the sampler - important for data shuffling!
        sampler.set_epoch(ep)
        
        # Make sure all processes are in sync before starting epoch
        safe_barrier(timeout=10.0)
        
        # Set model to training mode
        model.train()
        
        # Process batches
        total_batches = 0
        total_images = 0
        epoch_loss = 0.0
        epoch_correct = 0
        
        # Create CUDA events for timing (for the first epoch)
        if ep == 0 and torch.cuda.is_available():
            data_wait_start = torch.cuda.Event(enable_timing=True)
            data_wait_end = torch.cuda.Event(enable_timing=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            
            # Track stall times for later analysis
            data_wait_times = []
            compute_times = []
            total_times = []
        
        # Keep track of errors
        comm_errors = 0
        
        # Calculate exact number of batches per epoch for each rank
        num_batches_per_epoch = len(dataloader)
        
        # Process only a fixed number of batches per epoch per rank
        for batch_idx, (data, labels) in enumerate(dataloader):
            if batch_idx >= num_batches_per_epoch:
                break
                
            try:
                # Start data wait timing (for first epoch timing)
                if ep == 0 and torch.cuda.is_available():
                    data_wait_start.record()
                
                # Start a named range for profiling data loading
                with record_function("data_loading"):
                    # Move tensors to the correct device
                    data = data.to(device)
                    labels = labels.to(device)
                
                # End data wait timing and start compute timing
                if ep == 0 and torch.cuda.is_available():
                    data_wait_end.record()
                    compute_start.record()
                
                # Training step with profiling
                with record_function("training_step"):
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    with record_function("forward"):
                        outputs = model(data)
                        loss = criterion(outputs, labels)
                    
                    # Backward pass
                    with record_function("backward"):
                        loss.backward()
                    
                    # Optimization
                    with record_function("optimizer"):
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                
                # End compute timing
                if ep == 0 and torch.cuda.is_available():
                    compute_end.record()
                    
                    # Synchronize to get accurate times
                    torch.cuda.synchronize()
                    
                    # Calculate times in milliseconds
                    data_wait_time = data_wait_start.elapsed_time(data_wait_end)
                    compute_time = compute_start.elapsed_time(compute_end)
                    total_time = data_wait_time + compute_time
                    
                    # Store times
                    data_wait_times.append(data_wait_time)
                    compute_times.append(compute_time)
                    total_times.append(total_time)
                    
                    # Log times for the first few batches
                    if len(data_wait_times) <= 5 and rank == 0:
                        print(f"Batch {batch_idx} timing - Data loading: {data_wait_time:.2f}ms, Computing: {compute_time:.2f}ms")
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                
                # Update counters
                total_batches += 1
                batch_size = len(data)
                total_images += batch_size
                epoch_loss += loss.item() * batch_size
                epoch_correct += batch_correct
                
                # Step the profiler
                if profiler is not None and ep < args.profile_epochs:
                    profiler.step()
                
            except Exception as e:
                print(f"Error processing batch at rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                comm_errors += 1
                
                # If too many errors, exit the loop
                if comm_errors > 5:
                    print(f"Rank {rank}: Too many errors ({comm_errors}), stopping epoch early")
                    break
        
        # Print GPU stall time analysis after first epoch
        if ep == 0 and torch.cuda.is_available() and rank == 0 and len(data_wait_times) > 0:
            # Calculate statistics
            avg_data_wait = sum(data_wait_times) / len(data_wait_times)
            avg_compute = sum(compute_times) / len(compute_times)
            avg_total = sum(total_times) / len(total_times)
            
            # Calculate stall percentage (data wait time relative to total processing time)
            stall_percentage = (avg_data_wait / avg_total) * 100
            
            print("\n=== GPU Stall Time Analysis ===")
            print(f"Average data loading time: {avg_data_wait:.2f}ms")
            print(f"Average compute time: {avg_compute:.2f}ms")
            print(f"Average total time per batch: {avg_total:.2f}ms")
            print(f"GPU stall percentage: {stall_percentage:.2f}%")
            print(f"Theoretical maximum throughput: {1000/avg_total:.2f} batches/second")
            
            # Save timing data to a CSV file
            if len(data_wait_times) > 0 and args.profile:
                import csv
                with open(f"{args.profile_folder}/timing_data.csv", "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Batch", "Data_Wait_Time_ms", "Compute_Time_ms", "Total_Time_ms", "Stall_Percentage"])
                    for i in range(len(data_wait_times)):
                        stall_pct = (data_wait_times[i] / total_times[i]) * 100
                        writer.writerow([i, data_wait_times[i], compute_times[i], total_times[i], stall_pct])
        
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
        print(f"  - Total images: {total_images}")
        print(f"  - Loss: {avg_loss:.4f}")
        print(f"  - Accuracy: {avg_acc:.2f}%")
        print(f"  - Epoch time: {epoch_time:.2f} seconds")
        
        if total_images > 0:
            print(f"  - Images/second: {total_images / epoch_time:.2f}")
    
    # Cleanup profiler
    if profiler is not None:
        profiler.stop()
    
    # Record end time
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate overall training metrics
    overall_avg_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
    overall_avg_acc = 100 * total_train_correct / total_train_samples if total_train_samples > 0 else 0
    
    # if rank == 0:
    print("\n=== Distributed Training Complete ===")
    print(f"Overall Training Loss: {overall_avg_loss:.4f}")
    print(f"Overall Training Accuracy: {overall_avg_acc:.2f}%")
    print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    if args.profile:
        print(f"Profiling results saved to {args.profile_folder}")
        print(f"View with: tensorboard --logdir={args.profile_folder}")

    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
 