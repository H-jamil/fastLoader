import numpy as np
from .core import _fastloader
from typing import Tuple, List

class DataLoader:
    def __init__(self,
                 dataset_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4):
        """
        Initialize the data loader
        """
        self.prefetcher = _fastloader.Prefetcher(
            dataset_path,
            batch_size,
            num_workers
        )
        
    def __iter__(self):
        return self
        
    def __next__(self) -> Tuple[List[bytes], List[int]]:
        """
        Get next batch of data
        Returns:
            Tuple of (data, labels)
        """
        try:
            batch = self.prefetcher.get_next_batch()
            return batch.data, batch.labels
        except RuntimeError:
            raise StopIteration