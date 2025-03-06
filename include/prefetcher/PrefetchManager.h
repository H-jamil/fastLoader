#ifndef DIST_DATALOADER_PREFETCH_MANAGER_H
#define DIST_DATALOADER_PREFETCH_MANAGER_H

#include <torch/torch.h>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <pybind11/pybind11.h>
#include "../storage/StorageBackend.h"
#include "../distributed/DistributedManager.h"
#include "BufferManager.h"

namespace py = pybind11;

struct PrefetchItem {
    std::vector<torch::Tensor> data;
    std::vector<int> labels;
};

class PrefetchManager {
public:
    PrefetchManager(StorageBackend* storage_backend,
                   DistributedManager* dist_manager,
                   size_t prefetch_factor = 2,
                   int num_workers = 2,
                   int batch_size = 32);
    ~PrefetchManager();

    void start_prefetching();
    void stop_prefetching();
    PrefetchItem get_next_batch();
    int get_num_samples() const;
    py::object __iter__();
    py::object __next__();
    void debug_info() const;

private:
    void prefetch_worker();
    torch::Tensor load_image(int idx);

    StorageBackend* storage_backend;
    DistributedManager* dist_manager;
    std::unique_ptr<BufferManager> buffer_manager;
    
    size_t prefetch_factor;
    int num_workers;
    bool should_stop;
    std::vector<std::thread> worker_threads;
    
    std::mutex queue_mutex;
    std::condition_variable queue_not_empty;
    std::condition_variable queue_not_full;
    std::queue<PrefetchItem> prefetch_queue;
};

#endif // DIST_DATALOADER_PREFETCH_MANAGER_H
