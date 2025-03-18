#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import glob
import socket
from pathlib import Path


def print_tensor_info(tensor, name):
    """Print information about a PyTorch tensor"""
    if tensor is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")


class ImageNetDataset(Dataset):
    """Custom dataset for loading images"""
    
    def __init__(self, root_dir, transform=None):
        """Initialize the dataset with root directory and optional transform"""
        self.root_dir = root_dir
        self.transform = transform
        
        # Find all image files and their classes
        self.samples = []
        self.class_to_idx = {}
        
        # Assuming the structure is root/class_name/image.jpg
        class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Create class index mapping
        for idx, class_name in enumerate(sorted(class_dirs)):
            self.class_to_idx[class_name] = idx
            
            # Add each image in this class
            class_path = os.path.join(root_dir, class_name)
            for img_path in glob.glob(os.path.join(class_path, "*.JPEG")) + \
                          glob.glob(os.path.join(class_path, "*.jpg")) + \
                          glob.glob(os.path.join(class_path, "*.jpeg")) + \
                          glob.glob(os.path.join(class_path, "*.png")):
                self.samples.append((img_path, idx))
        
        print(f"Found {len(self.samples)} samples across {len(self.class_to_idx)} classes in {root_dir}")
    
    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load and transform an image at the given index"""
        img_path, label = self.samples[idx]
        
        try:
            # Load image using PIL
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            
            # Apply transformations if specified
            if self.transform:
                img = self.transform(img)
            else:
                # Default conversion to tensor
                img = transforms.ToTensor()(img)
            
            return img, label
        except Exception as e:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(f"Error loading image {img_path}: {e}")
            # Return a small black image and the label in case of errors
            return torch.zeros((3, 32, 32)), label


def custom_collate(batch):
    """
    Custom collate function that handles tensors of different sizes.
    Returns a list of tensors instead of stacked tensors.
    """
    images = [item[0] for item in batch]  # Get all images
    labels = [item[1] for item in batch]  # Get all labels
    
    return images, torch.tensor(labels)


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


def setup_distributed(rank, world_size, master_addr='10.52.0.178', master_port='29500'):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Check if CUDA is available and choose backend accordingly
    if torch.cuda.is_available():
        backend = 'nccl'  # NCCL for GPU
    else:
        backend = 'gloo'  # Gloo for CPU
    
    print(f"Rank {rank}: Using {backend} backend on {socket.gethostname()}")
    
    # Initialize process group with TCP initialization
    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://{master_addr}:{master_port}',
        rank=rank,
        world_size=world_size
    )
    print(f"Initialized process {rank} of {world_size} on {socket.gethostname()}")


def main():
    """Main function to test PyTorch Distributed Data Loading"""
    parser = argparse.ArgumentParser(description='Test PyTorch Distributed Data Loading')
    parser.add_argument('--data_path', type=str, default='/mnt/cephfs/subset/train/',
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for loading')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs to run')
    parser.add_argument('--target_size', type=str, default='224,224',
                        help='Target size for resizing images, format: width,height')
    parser.add_argument('--prefetch_factor', type=int, default=8,
                        help='Number of batches loaded in advance by each worker')
    parser.add_argument('--master_addr', type=str, default='10.52.0.178',
                        help='IP address of the master node')
    parser.add_argument('--master_port', type=str, default='29500',
                        help='Port of the master node')
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Get MPI rank and world size from environment variables
    # Check for OpenMPI environment variables
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
        world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
        print(f"Using OpenMPI environment: rank={rank}, world_size={world_size}")
    # Check for MPICH environment variables
    elif 'PMI_RANK' in os.environ:
        rank = int(os.environ.get('PMI_RANK', '0'))
        world_size = int(os.environ.get('PMI_SIZE', '1'))
        print(f"Using MPICH environment: rank={rank}, world_size={world_size}")
    # Check for any custom RANK and WORLD_SIZE environment variables
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ.get('RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f"Using custom environment variables: rank={rank}, world_size={world_size}")
    # Default to single process
    else:
        rank = 0
        world_size = 1
        print("No distributed environment detected, running in single process mode")

    # Set up distributed backend if we have multiple processes
    if world_size > 1:
        setup_distributed(rank, world_size, args.master_addr, args.master_port)
    
    # Create transforms for image preprocessing
    transform = transforms.ToTensor()
    
    # Start timing
    start_time = time.time()
    
    # Create dataset
    dataset = ImageNetDataset(args.data_path, transform=transform)
    
    # Create distributed sampler if we're in a distributed setting
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=False
        )
    else:
        sampler = None  # Use random sampling for single process
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if we're not using a sampler
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=custom_collate  # Use our custom collate function
    )
    
    # Calculate initialization time
    init_time = time.time() - start_time
    
    # Print information on rank 0 or if we're in single process mode
    if rank == 0 or world_size == 1:
        print(f"=== Testing PyTorch Distributed Data Loading ===")
        print(f"Dataset scan time: {init_time:.2f} seconds")
        print(f"Global number of samples: {len(dataset)}")
        print(f"Number of classes: {len(dataset.class_to_idx)}")
        print(f"Running with {world_size} processes")
        print(f"Target image size: {target_size}")
    
    # Run multiple epochs
    for ep in range(args.num_epochs):
        if rank == 0 or world_size == 1:
            print(f"\n=== Starting Epoch {ep+1} ===")
        
        # Set epoch for sampler - ensures different shuffling each epoch
        if sampler is not None:
            sampler.set_epoch(ep)
        
        # Adjust workers based on epoch (example) - matching FastLoader example
        if ep == 2:
            new_num_workers = 12
            if rank == 0 or world_size == 1:
                print(f"Increasing number of workers to {new_num_workers}")
            
            # Create a new dataloader with more workers
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=new_num_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True if new_num_workers > 0 else False,
                collate_fn=custom_collate  # Use our custom collate function
            )
        
        # Process batches
        epoch_start = time.time()
        total_batches = 0
        total_images = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            try:
                # Here we resize the data similar to FastLoader
                data_batch = preprocess_batch(data, target_size=target_size)
                
                total_batches += 1
                total_images += len(data)
                
                # Print progress
                if total_batches % 10 == 0 and (rank == 0 or world_size == 1):
                    print(f"Processed {total_batches} batches, {total_images} images")
                
                # Print first batch info (for debugging)
                if total_batches == 1 and (rank == 0 or world_size == 1):
                    print_tensor_info(data_batch, "First image batch")
                    print_tensor_info(labels, "First label batch")
                
            except Exception as e:
                print(f"Error processing batch at rank {rank}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print performance metrics for each rank
        print(f"Process {rank} Performance:")
        print(f"  - Total batches: {total_batches}")
        print(f"  - Total images: {total_images}")
        print(f"  - Epoch time: {epoch_time:.2f} seconds")
        
        if total_images > 0:
            print(f"  - Images/second: {total_images / epoch_time:.2f}")
        
        # Add barrier to synchronize all processes before next epoch
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
    
    # Clean up distributed resources
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0 or world_size == 1:
        print("\n=== PyTorch Distributed Test Complete ===")
    end_time = time.time()

    print(f"-----start time {start_time} and end time {end_time} total duration {end_time-start_time}")


if __name__ == "__main__":
    main()
