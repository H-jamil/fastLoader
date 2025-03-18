# Roadmap: PyTorch-Compatible Distributed Prefetching DataLoader

## Introduction

This document outlines the roadmap for creating a Python interface to our C++ prefetching implementation. The goal is to develop a PyTorch-compatible DataLoader that leverages our high-performance C++ prefetching backend while providing dynamic thread management capabilities through a Python API.

## Motivation

The standard PyTorch DataLoader has limitations in distributed environments, particularly with dynamic resource adaptation. Our implementation will:

1. Provide high-performance data loading using our optimized C++ implementation
2. Allow dynamic adjustment of prefetching threads at runtime
3. Integrate seamlessly with PyTorch training loops
4. Support distributed training environments

## High-Level Architecture

```
+-------------------+      +------------------------+      +-------------------+
|                   |      |                        |      |                   |
|  Python API       |<---->|  C++ Extension Module  |<---->|  C++ Prefetching  |
|  (PyTorch-like)   |      |  (pybind11/cython)     |      |  Implementation   |
|                   |      |                        |      |                   |
+-------------------+      +------------------------+      +-------------------+
```

## Detailed Roadmap

### Phase 1: C++ Extension Module (Weeks 1-2)

1. **Choose binding technology**
   - Select pybind11 for its seamless C++ integration and ease of use
   - Set up build environment to support Python extension module

2. **Define core C++ interface**
   - Create a `PyTorchPrefetcher` wrapper class that encapsulates:
     - `FileSystemBackend`
     - `DistributedManager`
     - `PrefetchManager`
   - Expose the following core methods:
     - `initialize(data_path, batch_size, num_workers, prefetch_factor)`
     - `get_next_batch()`
     - `set_num_workers(num)`
     - `start_prefetching()`
     - `stop_prefetching()`
     - `reset()`

3. **Implement tensor conversion**
   - Create utilities to convert between C++ tensors and PyTorch tensors
   - Ensure zero-copy or minimal-copy operations where possible
   - Properly handle GPU tensor conversions if needed

4. **Add basic error handling**
   - Implement exception handling that properly propagates from C++ to Python
   - Ensure resource cleanup on errors

### Phase 2: Python Interface (Weeks 3-4)

1. **Create PyTorch DataLoader-compatible class**
   - Implement a `FastDataLoader` class that mimics PyTorch's DataLoader interface:
     - `__iter__()`: Returns self after setting up prefetching
     - `__next__()`: Gets the next batch from C++ backend
     - Support same initialization parameters as PyTorch's DataLoader where relevant

2. **Implement adaptive thread management**
   - Add methods for runtime thread management:
     - `set_num_workers(num)`: Dynamically adjust thread count
     - `get_performance_stats()`: Return performance metrics
     - `auto_tune()`: Automatically adjust thread count based on system load

3. **Add PyTorch compatibility features**
   - Support for custom samplers
   - Support for data transforms
   - Add collate_fn functionality

4. **Implement distributed training support**
   - Add compatibility with PyTorch DistributedDataParallel (DDP)
   - Support for sharding data across nodes
   - Implement proper process group synchronization

### Phase 3: Build System and Packaging (Week 5)

1. **Extend CMake configuration**
   - Add targets to build the Python extension module
   - Integrate with pybind11
   - Ensure proper linking with existing C++ libraries

2. **Create Python package**
   - Set up setup.py for proper installation
   - Configure package for pip installation
   - Create proper documentation

3. **Implement CI/CD**
   - Add automated tests for the Python interface
   - Create build pipeline
   - Add performance benchmarks

### Phase 4: Optimization and Advanced Features (Weeks 6-8)

1. **Performance optimization**
   - Profile and optimize critical paths
   - Minimize GIL contention
   - Optimize memory usage and reduce copies

2. **Advanced thread management**
   - Implement adaptive algorithms for worker adjustment
   - Add system monitoring for resource-aware scaling
   - Support priority-based prefetching

3. **Extended format support**
   - Add support for different file formats and datasets
   - Create plugin system for custom data sources

4. **Monitoring and debugging tools**
   - Add visualization for prefetching performance
   - Create debugging utilities

## Implementation Details

### C++ Extension with pybind11

Here's an example of what the C++ extension module could look like:

```cpp
// prefetcher_python.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "prefetcher/PrefetchManager.h"
#include "storage/FileSystemBackend.h"
#include "distributed/DistributedManager.h"

namespace py = pybind11;

class PyTorchPrefetcher {
private:
    FileSystemBackend* backend;
    DistributedManager* dist_manager;
    PrefetchManager* prefetch_manager;
    int batch_size;
    bool initialized;

public:
    PyTorchPrefetcher() : backend(nullptr), dist_manager(nullptr), 
                         prefetch_manager(nullptr), batch_size(32), initialized(false) {}
    
    void initialize(const std::string& data_path, int batch_size, int num_workers, int prefetch_factor) {
        this->batch_size = batch_size;
        backend = new FileSystemBackend(data_path);
        dist_manager = new DistributedManager(backend->get_num_samples(), batch_size);
        prefetch_manager = new PrefetchManager(backend, dist_manager, prefetch_factor, num_workers, batch_size);
        initialized = true;
    }
    
    py::tuple get_next_batch() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        
        PrefetchItem batch = prefetch_manager->get_next_batch();
        if (batch.data.empty()) {
            return py::make_tuple(py::none(), py::none());
        }
        
        // Convert C++ tensors to PyTorch tensors
        std::vector<torch::Tensor> data_tensors;
        std::vector<int64_t> labels;
        
        for (size_t i = 0; i < batch.data.size(); i++) {
            data_tensors.push_back(batch.data[i]);
            labels.push_back(batch.labels[i]);
        }
        
        // Stack tensors into a batch
        auto data_batch = torch::stack(data_tensors);
        auto labels_batch = torch::tensor(labels);
        
        return py::make_tuple(data_batch, labels_batch);
    }
    
    void set_num_workers(int num) {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        prefetch_manager->set_num_workers(num);
    }
    
    void start_prefetching() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        prefetch_manager->start_prefetching();
    }
    
    void stop_prefetching() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        prefetch_manager->stop_prefetching();
    }
    
    void reset() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        dist_manager->reset_epoch();
    }
    
    ~PyTorchPrefetcher() {
        if (prefetch_manager) delete prefetch_manager;
        if (dist_manager) delete dist_manager;
        if (backend) delete backend;
    }
};

PYBIND11_MODULE(fastloader, m) {
    m.doc() = "FastLoader: High-performance data loading with adaptive prefetching";
    
    py::class_<PyTorchPrefetcher>(m, "Prefetcher")
        .def(py::init<>())
        .def("initialize", &PyTorchPrefetcher::initialize, 
             py::arg("data_path"), py::arg("batch_size")=32, 
             py::arg("num_workers")=4, py::arg("prefetch_factor")=2)
        .def("get_next_batch", &PyTorchPrefetcher::get_next_batch)
        .def("set_num_workers", &PyTorchPrefetcher::set_num_workers)
        .def("start_prefetching", &PyTorchPrefetcher::start_prefetching)
        .def("stop_prefetching", &PyTorchPrefetcher::stop_prefetching)
        .def("reset", &PyTorchPrefetcher::reset);
}
```

### Python DataLoader Interface

Example of the Python interface:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import fastloader  # Your C++ extension module

class FastDataLoader:
    def __init__(self, data_path, batch_size=32, num_workers=4, 
                 prefetch_factor=2, shuffle=True, drop_last=False):
        self.prefetcher = fastloader.Prefetcher()
        self.prefetcher.initialize(data_path, batch_size, num_workers, prefetch_factor)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        
    def __iter__(self):
        self.prefetcher.reset()  # Reset for new epoch
        self.prefetcher.start_prefetching()
        return self
    
    def __next__(self):
        data, labels = self.prefetcher.get_next_batch()
        if data is None:  # End of epoch
            self.prefetcher.stop_prefetching()
            self.epoch += 1
            raise StopIteration
        return data, labels
    
    def set_num_workers(self, num):
        """Dynamically adjust the number of worker threads"""
        self.prefetcher.set_num_workers(num)
    
    def auto_tune(self):
        """Auto-tune thread count based on system metrics"""
        # Implement auto-tuning logic based on system load, memory usage, etc.
        pass
```

### Integration with PyTorch Training Loop

Example usage in a training loop:

```python
import torch
import torch.nn as nn
from fastloader import FastDataLoader

# Define model, loss function, optimizer
model = YourModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create data loader
dataloader = FastDataLoader(
    data_path="/path/to/data",
    batch_size=32,
    num_workers=4,
    prefetch_factor=2
)

# Training loop
for epoch in range(10):
    for data, target in dataloader:
        # Move data to device (GPU)
        data, target = data.to('cuda'), target.to('cuda')
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Dynamic worker adjustment based on system load
        if some_condition:
            dataloader.set_num_workers(8)  # Increase workers
        elif another_condition:
            dataloader.set_num_workers(2)  # Decrease workers
```

### CMake Configuration

Example CMake adjustments:

```cmake
# Add pybind11
find_package(pybind11 REQUIRED)

# Add PyTorch extension module
pybind11_add_module(fastloader prefetcher_python.cpp)

# Link against your existing libraries
target_link_libraries(fastloader PRIVATE 
    prefetcher_lib
    storage_lib
    distributed_lib
    ${TORCH_LIBRARIES}
    ${PYTHON_LIBRARIES}
)
```

## Challenges and Considerations

1. **Memory Management**
   - Ensuring proper ownership of tensors between C++ and Python
   - Using shared memory where possible to avoid copies
   - Handling memory pressure in distributed environments

2. **Thread Safety**
   - Managing Python's GIL (Global Interpreter Lock)
   - Releasing the GIL during computationally intensive operations
   - Ensuring thread-safe access to shared resources

3. **Distributed Training Support**
   - Handling PyTorch's DistributedSampler and process groups
   - Ensuring compatibility with PyTorch DDP (Distributed Data Parallel)
   - Correctly sharding data across nodes

4. **Error Handling**
   - Propagating C++ exceptions to Python properly
   - Providing meaningful error messages
   - Ensuring cleanup on failures

5. **Performance Monitoring**
   - Adding metrics to track prefetching performance
   - Implementing adaptive strategies for worker thread management
   - Creating visualizations for performance analysis

## Next Steps

1. Set up project structure with CMake support for pybind11
2. Implement minimal C++ extension module with basic functionality
3. Create Python wrapper with DataLoader-compatible interface
4. Add dynamic thread management
5. Test with real PyTorch training loops
6. Extend with distributed training support
7. Optimize and benchmark performance
8. Develop advanced auto-tuning capabilities 