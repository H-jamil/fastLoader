import torch
import os
import time
from fastloader import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import io

class ImageNetDataLoader:
    def __init__(self, 
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: int = 224):
        """
        Initialize ImageNet data loader
        """
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize the fast loader
        self.loader = DataLoader(
            dataset_path=data_dir,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.iterator = iter(self.loader)
        
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self
        
    def __next__(self):
        try:
            # Get raw data from loader
            raw_images, labels = next(self.iterator)
            
            # Convert raw images to PIL Images and apply transforms
            processed_images = []
            for raw_img in raw_images:
                try:
                    # Convert raw bytes to PIL Image
                    img = Image.open(io.BytesIO(raw_img))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Apply transforms
                    tensor_img = self.transform(img)
                    processed_images.append(tensor_img)
                except Exception as e:
                    continue
                
            if not processed_images:
                raise RuntimeError("No images were successfully processed in batch")
                
            # Stack all images into a batch
            batch_tensor = torch.stack(processed_images)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            return batch_tensor, labels_tensor
            
        except StopIteration:
            raise StopIteration

def main():
    # Set your ImageNet training directory
    data_dir = "/home/cc/train/subset/train"
    
    # Configuration parameters
    batch_size = 32
    num_workers = 4
    image_size = 224
    
    print("Creating DataLoader...")
    # Create the data loader
    data_loader = ImageNetDataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size
    )
    print("DataLoader created")
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Iterate over batches to test I/O
    start_time = time.time()
    try:
        for batch_idx, (images, labels) in enumerate(data_loader):
            # Move data to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if batch_idx % 5 == 0:
                print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
                
        print(f"Took {time.time()-start_time} seconds")
            
    except Exception as e:
        print(f"Error during iteration: {e}")
    finally:
        print("Testing completed")

if __name__ == "__main__":
    main()