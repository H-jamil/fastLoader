import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
import argparse
import time

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Scan directory structure
        class_idx = 0
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            self.class_to_idx[class_name] = class_idx
            self.idx_to_class[class_idx] = class_name
            
            for img_name in os.listdir(class_path):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, class_idx))
            
            class_idx += 1
            
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

def train(rank, world_size, dataset_path):
    # Set up the distributed environment
    # Note: MASTER_ADDR and MASTER_PORT are set in main() from command line args
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set up GPU device - use local rank instead of global rank
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    print(f"Node {rank} using device: {device}")
    
    # Create dataset and distributed sampler
    dataset = ImageFolderDataset(dataset_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create data loader
    batch_size = 32
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        # Set epoch for proper shuffling
        sampler.set_epoch(epoch)
        
        print(f"Node {rank}, Starting epoch {epoch+1}")
        
        # Track epoch statistics
        epoch_start_time = time.time()
        total_images = 0
        
        try:
            for batch_idx, (images, labels) in enumerate(loader):
                # Move data to GPU
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Update statistics
                total_images += images.size(0)
                
                # Print progress every 20 batches
                if batch_idx % 20 == 0:
                    current_time = time.time()
                    images_per_second = total_images / (current_time - epoch_start_time)
                    print(f"Node {rank}, Epoch {epoch+1}, "
                          f"Batch {batch_idx}, "
                          f"Batch size: {images.size(0)}, "
                          f"Images/sec: {images_per_second:.2f}")
                
        except Exception as e:
            print(f"Error in training loop at rank {rank}, epoch {epoch+1}: {str(e)}")
            continue
        
        # Print epoch summary
        epoch_duration = time.time() - epoch_start_time
        print(f"\nNode {rank}, Epoch {epoch+1} Summary:")
        print(f"  Duration: {epoch_duration:.2f} seconds")
        print(f"  Total Images: {total_images}")
        print(f"  Average Speed: {total_images/epoch_duration:.2f} images/second")
        
        # Synchronize at epoch end
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    # Cleanup
    dist.destroy_process_group()

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
    
    # Start training
    train(args.node_rank, args.world_size, args.dataset_path)

if __name__ == "__main__":
    main() 