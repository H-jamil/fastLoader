#!/usr/bin/env python3
import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from pathlib import Path


def print_tensor_info(tensor, name):
    """Print information about a PyTorch tensor"""
    if tensor is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")


class ImageNetDataset(Dataset):
    """Custom dataset for loading images similar to what FastLoader loads"""
    
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
            print(f"Error loading image {img_path}: {e}")
            # Return a small black image and the label in case of errors
            return torch.zeros((3, 32, 32)), label


def main():
    """Main function to test PyTorch DataLoader"""
    parser = argparse.ArgumentParser(description='Test PyTorch DataLoader with ImageNet')
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
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches loaded in advance by each worker')
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    # Create transforms for image preprocessing
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    
    # Start timing
    start_time = time.time()
    
    # Create dataset
    dataset = ImageNetDataset(args.data_path, transform=transform)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Calculate initialization time
    init_time = time.time() - start_time
    
    # Print information
    print(f"=== Testing PyTorch DataLoader with Dataset ===")
    print(f"Dataset scan time: {init_time:.2f} seconds")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_to_idx)}")
    print(f"Running with {args.num_workers} worker processes")
    print(f"Target image size: {target_size}")
    
    # Run multiple epochs
    for ep in range(args.num_epochs):
        print(f"\n=== Starting Epoch {ep+1} ===")
        
        # Adjust workers based on epoch (example)
        if ep == 2:
            new_num_workers = 12
            print(f"Increasing number of workers to {new_num_workers}")
            
            # Create a new dataloader with more workers
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=new_num_workers,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=True,
            )
        
        # Process batches
        epoch_start = time.time()
        total_batches = 0
        total_images = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            try:
                total_batches += 1
                total_images += len(data)
                
                # Print progress
                if total_batches % 10 == 0:
                    print(f"Processed {total_batches} batches, {total_images} images")
                
                # Print first batch info (for debugging)
                if total_batches == 1:
                    print_tensor_info(data, "First image batch")
                    print_tensor_info(labels, "First label batch")
                
                # Simulate some processing (could be a network forward pass)
                # This small sleep simulates computation time to make comparison fair
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        
        # Print performance metrics
        print(f"Performance:")
        print(f"  - Total batches: {total_batches}")
        print(f"  - Total images: {total_images}")
        print(f"  - Epoch time: {epoch_time:.2f} seconds")
        
        if total_images > 0:
            print(f"  - Images/second: {total_images / epoch_time:.2f}")
    
    print("\n=== PyTorch DataLoader Test Complete ===")


if __name__ == "__main__":
    main() 