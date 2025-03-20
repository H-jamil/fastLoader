#ifndef DIST_DATALOADER_PREFETCH_MANAGER_H
#define DIST_DATALOADER_PREFETCH_MANAGER_H

#include <torch/torch.h>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>
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
                   int initial_num_workers = 2,
                   int batch_size = 32,
                   bool enable_preprocessing = false,
                   std::pair<int, int> target_size = {224, 224});
    ~PrefetchManager();

    // Core functionality
    void start_prefetching();
    void stop_prefetching();
    PrefetchItem get_next_batch();
    int get_num_samples() const;
    py::object __iter__();
    py::object __next__();
    
    // Dynamic worker management
    void set_num_workers(int new_num_workers);
    int get_num_workers() const;
    void set_prefetch_factor(size_t new_factor);
    size_t get_prefetch_factor() const;
    
    // Preprocessing controls
    void set_preprocessing(bool enable, std::pair<int, int> target_size = {224, 224});
    bool is_preprocessing_enabled() const;
    std::pair<int, int> get_target_size() const;
    
    // Debug and monitoring
    void debug_info() const;
    size_t get_queue_size() const;
    bool is_prefetching() const;

private:
    void prefetch_worker();
    torch::Tensor load_image(int idx);
    torch::Tensor preprocess_tensor(const torch::Tensor& tensor);
    void adjust_worker_count(int new_count);
    void cleanup_workers();

    StorageBackend* storage_backend;
    DistributedManager* dist_manager;
    std::unique_ptr<BufferManager> buffer_manager;
    
    size_t prefetch_factor;
    std::atomic<int> num_workers;
    std::atomic<bool> should_stop;
    std::atomic<bool> is_running;
    
    // Preprocessing options
    std::atomic<bool> preprocessing_enabled;
    std::pair<int, int> preprocessing_target_size;
    mutable std::mutex preprocessing_mutex;
    
    std::vector<std::thread> worker_threads;
    mutable std::mutex worker_mutex;
    
    mutable std::mutex queue_mutex;
    std::condition_variable queue_not_empty;
    std::condition_variable queue_not_full;
    std::queue<PrefetchItem> prefetch_queue;
};

#endif // DIST_DATALOADER_PREFETCH_MANAGER_H
