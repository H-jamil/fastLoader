#ifndef DIST_DATALOADER_PREFETCH_MANAGER_H
#define DIST_DATALOADER_PREFETCH_MANAGER_H

#include <torch/torch.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include "../storage/StorageBackend.h"
#include "../distributed/DistributedManager.h"
#include <pybind11/pybind11.h>

struct PrefetchItem {
    // Updated to hold a batch of data instead of individual items
    std::vector<torch::Tensor> data;
    std::vector<int> labels;
};

class PrefetchManager {
public:
    PrefetchManager(StorageBackend* storage_backend,
                   DistributedManager* dist_manager,
                   int prefetch_factor,
                   int num_workers,
                   int batch_size);
    ~PrefetchManager();

    void start_prefetching();
    void stop_prefetching();
    int get_num_samples() const;
    
    // Modified to return a batch instead of a single item
    PrefetchItem get_next_batch();
    
    // Iterator methods for Python
    pybind11::object __iter__();
    pybind11::object __next__();

private:
    StorageBackend* storage_backend;
    DistributedManager* dist_manager;
    int prefetch_factor;
    int num_workers;
    int batch_size;
    bool should_stop;
    
    std::vector<std::thread> worker_threads;
    std::queue<PrefetchItem> prefetch_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_not_full;
    std::condition_variable queue_not_empty;
    
    void prefetch_worker();
    torch::Tensor load_image(int idx);
};

#endif