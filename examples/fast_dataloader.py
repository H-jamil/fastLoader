#!/usr/bin/env python3
import time
import torch
from torch.utils.data import Dataset, DataLoader
from fastloader import Prefetcher

class FastDataLoader:
    """A PyTorch DataLoader-like class that uses our C++ prefetching backend.
    
    This class mimics the interface of PyTorch's DataLoader but uses our
    high-performance C++ prefetching implementation under the hood.
    """
    
    def __init__(self, data_path, batch_size=32, num_workers=4, 
                 prefetch_factor=2, shuffle=True, drop_last=False):
        """Initialize the FastDataLoader.
        
        Args:
            data_path (str): Path to the dataset
            batch_size (int): Batch size for each iteration
            num_workers (int): Number of worker threads for prefetching
            prefetch_factor (int): Number of batches to prefetch ahead
            shuffle (bool): Whether to shuffle the data at each epoch
            drop_last (bool): Whether to drop the last incomplete batch
        """
        self.prefetcher = Prefetcher()
        self.prefetcher.initialize(data_path, batch_size, num_workers, prefetch_factor)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        
        # Get dataset information
        self.dataset_info = self.prefetcher.get_dataset_info()
        self.length = self.dataset_info['num_samples'] // batch_size
        if not drop_last and self.dataset_info['num_samples'] % batch_size > 0:
            self.length += 1
            
        # Get MPI info
        self.rank = self.prefetcher.get_rank()
        self.world_size = self.prefetcher.get_world_size()
        
    def __iter__(self):
        """Reset for a new epoch and return iterator."""
        self.prefetcher.reset_epoch()
        self.prefetcher.start_prefetching()
        return self
    
    def __next__(self):
        """Get the next batch of data."""
        data, labels = self.prefetcher.get_next_batch()
        if data is None:  # End of epoch
            self.prefetcher.stop_prefetching()
            self.epoch += 1
            raise StopIteration
        return data, labels
    
    def __len__(self):
        """Return the number of batches in the dataset."""
        return self.length
    
    def set_num_workers(self, num):
        """Dynamically adjust the number of worker threads."""
        self.prefetcher.set_num_workers(num)
    
    def get_performance_stats(self):
        """Return performance statistics (to be implemented)."""
        # This would return metrics like throughput, latency, etc.
        return {}
    
    def auto_tune(self):
        """Auto-tune thread count based on system metrics.
        
        This is a placeholder for future implementation.
        """
        # This would monitor system load and adjust workers automatically
        pass


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test FastDataLoader')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')
    args = parser.parse_args()
    
    # Create dataloader
    dataloader = FastDataLoader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Print information
    print(f"Dataset has {dataloader.dataset_info['num_samples']} samples")
    print(f"Dataset has {dataloader.dataset_info['num_classes']} classes")
    print(f"DataLoader will yield {len(dataloader)} batches per epoch")
    
    # Process one epoch
    start_time = time.time()
    for i, (data, labels) in enumerate(dataloader):
        # Just print progress every 10 batches
        if i % 10 == 0:
            print(f"Batch {i}/{len(dataloader)}: data shape={data.shape}, labels shape={labels.shape}")
            
        # Example of dynamic thread adjustment
        if i == len(dataloader) // 2:
            print(f"Adjusting workers to 8 for second half of epoch")
            dataloader.set_num_workers(8)
            
    # Calculate elapsed time
    elapsed = time.time() - start_time
    print(f"Epoch completed in {elapsed:.2f} seconds") 