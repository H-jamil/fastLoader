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
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import glob
from PIL import Image


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


def get_rank_and_world_size():
    """Get the current process rank and world size"""
    if not dist.is_initialized():
        return 0, 1
    
    return dist.get_rank(), dist.get_world_size()


def main():
    """Main function to test PyTorch distributed ResNet50 training"""
    parser = argparse.ArgumentParser(description='Train ResNet50 with PyTorch Distributed Data Parallel')
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
    # We need to call setup_distributed before creating DistributedSampler
    # Assuming MASTER_ADDR and MASTER_PORT are set in the environment
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
        print(f"=== Distributed ResNet50 Training with PyTorch ===")
        print(f"Dataset scan time: {init_time:.2f} seconds")
        print(f"Global number of samples: {dataset_info['num_samples']}")
        print(f"Number of classes: {dataset_info['num_classes']}")
        print(f"Running with {world_size} processes")
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
        
        # Set the epoch for the sampler - important for data shuffling!
        sampler.set_epoch(ep)
        
        # Important: Make sure all processes are in sync before starting epoch
        safe_barrier(timeout=10.0)
        
        # Set model to training mode
        model.train()
        
        # Process batches
        total_batches = 0
        total_images = 0
        epoch_loss = 0.0
        epoch_correct = 0
        
        # Keep track of errors
        comm_errors = 0
        
        # Calculate exact number of batches per epoch for each rank
        num_batches_per_epoch = len(dataloader)
        
        # Ensure all ranks process the same exact number of batches
        # by broadcasting num_batches_per_epoch from rank 0
        max_batches_tensor = torch.tensor([num_batches_per_epoch], dtype=torch.long, device=device)
        dist.broadcast(max_batches_tensor, 0)
        max_batches_per_epoch = max_batches_tensor.item()
        
        if rank == 0:
            print(f"Processing exactly {max_batches_per_epoch} batches per process for this epoch")
        
        # Process only a fixed number of batches per epoch per rank
        for batch_idx, (data, labels) in enumerate(dataloader):
            if batch_idx >= max_batches_per_epoch:
                break
                
            try:
                # Periodically synchronize to make sure processes are still in step
                if batch_idx % 10 == 0:
                    safe_barrier(timeout=5.0)
                
                # Move tensors to the correct device
                data = data.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Check for NaN or inf in gradients
                if not check_gradients():
                    print(f"Warning: Rank {rank} detected NaN or inf in gradients at batch {batch_idx}. Skipping update.")
                    optimizer.zero_grad()  # Clear bad gradients
                else:
                    # Optimize
                    optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                
                # Update counters
                total_batches += 1
                batch_size = len(data)
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
                    print_tensor_info(data, "First image batch")
                    print_tensor_info(labels, "First label batch")
                
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
