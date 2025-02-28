// src/prefetcher/PrefetchManager.cpp
#include "../../include/prefetcher/PrefetchManager.h"
#include <torch/torch.h>
#include <stdexcept>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

PrefetchManager::PrefetchManager(StorageBackend* storage_backend,
                               DistributedManager* dist_manager,
                               int prefetch_factor,
                               int num_workers,
                               int batch_size)
    : storage_backend(storage_backend),
      dist_manager(dist_manager),
      prefetch_factor(prefetch_factor),
      num_workers(num_workers),
      batch_size(batch_size),
      should_stop(false) {
    
    if (!storage_backend || !dist_manager) {
        throw std::runtime_error("Invalid backend or distributed manager");
    }
}

PrefetchManager::~PrefetchManager() {
    stop_prefetching();
}

void PrefetchManager::start_prefetching() {
    should_stop = false;
    
    // Create worker threads
    for (int i = 0; i < num_workers; ++i) {
        worker_threads.emplace_back(&PrefetchManager::prefetch_worker, this);
    }
}

void PrefetchManager::stop_prefetching() {
    should_stop = true;
    queue_not_empty.notify_all();
    queue_not_full.notify_all();
    
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads.clear();
    
    // Clear queue
    std::queue<PrefetchItem>().swap(prefetch_queue);
}

void PrefetchManager::prefetch_worker() {
    while (!should_stop) {
        // Get next batch indices from distributed manager
        auto batch_indices = dist_manager->get_next_batch_indices();
        if (batch_indices.empty()) {
            continue;
        }
        
        // Process all batch indices before locking the queue
        std::vector<torch::Tensor> batch_data;
        std::vector<int> batch_labels;
        batch_data.reserve(batch_indices.size());
        batch_labels.reserve(batch_indices.size());
        
        for (int idx : batch_indices) {
            if (should_stop) break;
            
            try {
                // Load image with minimal preprocessing
                auto tensor = load_image(idx);
                int label = storage_backend->get_label_index(idx);
                
                // Add to local vectors
                batch_data.push_back(std::move(tensor));
                batch_labels.push_back(label);
            } catch (const std::exception& e) {
                std::cerr << "Error in prefetch worker: " << e.what() << std::endl;
            }
        }
        
        // Only lock once to add the whole batch to the queue
        if (!batch_data.empty() && !should_stop) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_not_full.wait(lock, [this]() {
                return prefetch_queue.size() < prefetch_factor || should_stop;
            });
            
            if (should_stop) continue;
            
            PrefetchItem item{std::move(batch_data), std::move(batch_labels)};
            prefetch_queue.push(std::move(item));
            queue_not_empty.notify_one();
        }
    }
}

torch::Tensor PrefetchManager::load_image(int idx) {
    const Sample& sample = storage_backend->get_sample(idx);
    std::vector<unsigned char> buffer(sample.size);
    storage_backend->fetch(idx, buffer.data());
    
    // Decode image using OpenCV with minimal preprocessing
    cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Failed to decode image: " + sample.path);
    }
    
    // Convert to tensor without additional preprocessing
    auto tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8).clone();
    
    // Ensure tensor is in the right format (HWC layout) but don't normalize/resize
    return tensor;
}

PrefetchItem PrefetchManager::get_next_batch() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    
    queue_not_empty.wait(lock, [this]() {
        return !prefetch_queue.empty() || should_stop;
    });
    
    if (should_stop) {
        throw std::runtime_error("PrefetchManager is stopped");
    }
    
    PrefetchItem batch = std::move(prefetch_queue.front());
    prefetch_queue.pop();
    queue_not_full.notify_one();
    
    return batch;
}

int PrefetchManager::get_num_samples() const {
    return storage_backend->get_num_samples();
}

py::object PrefetchManager::__iter__() {
    return py::cast(this);
}

py::object PrefetchManager::__next__() {
    try {
        // Get a batch directly from the queue
        PrefetchItem batch = get_next_batch();
        
        if (batch.data.empty()) {
            throw std::runtime_error("No valid images in batch");
        }
        
        // Stack tensors along the batch dimension
        auto data_tensor = torch::stack(batch.data);
        
        // Convert labels to long tensor
        std::vector<int64_t> long_labels(batch.labels.begin(), batch.labels.end());
        auto labels_tensor = torch::tensor(long_labels, torch::kLong);
        
        return py::make_tuple(data_tensor, labels_tensor);
    } catch (const std::exception& e) {
        std::cerr << "Error in __next__: " << e.what() << std::endl;
        throw py::stop_iteration();
    }
}