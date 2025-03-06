import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
from PIL import Image
import torchvision.transforms as transforms
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

class DistributedDataLoader:
    def __init__(self, dataset_path, batch_size, num_workers=4, prefetch_factor=4):
        # Initialize the distributed environment
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Create dataset
        self.dataset = ImageFolderDataset(dataset_path)
        
        # Create distributed sampler
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        # Create DataLoader with enhanced prefetching
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            drop_last=True  # Ensure consistent batch sizes
        )
        
        # Add performance tracking
        self.batch_times = []
        self.last_batch_time = None
        self.epoch_start_time = None
        self.epoch_total_images = 0
        
    def __iter__(self):
        self.last_batch_time = time.time()
        return iter(self.loader)
        
    def __len__(self):
        return len(self.loader)
        
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.batch_times = []  # Reset timing for new epoch
        self.epoch_start_time = time.time()
        self.epoch_total_images = 0
        
    def update_epoch_stats(self, batch_size):
        self.epoch_total_images += batch_size
        
    def get_epoch_stats(self):
        if self.epoch_start_time is None:
            return None
            
        epoch_duration = time.time() - self.epoch_start_time
        images_per_second = self.epoch_total_images / epoch_duration
        
        return {
            'epoch_duration': epoch_duration,
            'total_images': self.epoch_total_images,
            'images_per_second': images_per_second
        }

    def get_performance_metrics(self):
        if not self.batch_times:
            return None
        avg_batch_time = sum(self.batch_times) / len(self.batch_times)
        return {
            'avg_batch_time': avg_batch_time,
            'batches_per_second': 1.0 / avg_batch_time,
            'total_batches': len(self.batch_times)
        }
