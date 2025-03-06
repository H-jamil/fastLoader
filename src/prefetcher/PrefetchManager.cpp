// src/prefetcher/PrefetchManager.cpp
#include "../../include/prefetcher/PrefetchManager.h"
#include <torch/torch.h>
#include <stdexcept>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace py = pybind11;

PrefetchManager::PrefetchManager(StorageBackend* storage_backend,
                               DistributedManager* dist_manager,
                               size_t prefetch_factor,
                               int num_workers,
                               int batch_size)
    : storage_backend(storage_backend),
      dist_manager(dist_manager),
      prefetch_factor(prefetch_factor),
      num_workers(num_workers),
      should_stop(false) {
    // Initialize buffer manager with appropriate size
    size_t max_sample_size = 0;
    for (int i = 0; i < storage_backend->get_num_samples(); ++i) {
        max_sample_size = std::max(max_sample_size, storage_backend->get_file_size(i));
    }
    buffer_manager = std::make_unique<BufferManager>(max_sample_size * prefetch_factor, batch_size);
}

PrefetchManager::~PrefetchManager() {
    stop_prefetching();
}

void PrefetchManager::start_prefetching() {
    should_stop = false;
    worker_threads.clear();
    
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
    
    // Clear the queue
    std::queue<PrefetchItem>().swap(prefetch_queue);
}

void PrefetchManager::prefetch_worker() {
    while (!should_stop) {
        try {
            auto batch_indices = dist_manager->get_next_batch_indices();
            if (batch_indices.empty()) {
                // End of epoch
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_not_full.wait(lock, [this]() {
                    return prefetch_queue.size() < prefetch_factor || should_stop;
                });
                
                if (should_stop) break;
                
                PrefetchItem end_marker;  // Empty item
                prefetch_queue.push(std::move(end_marker));
                queue_not_empty.notify_one();
                break;
            }
            
            std::vector<torch::Tensor> batch_data;
            std::vector<int> batch_labels;
            batch_data.reserve(batch_indices.size());
            batch_labels.reserve(batch_indices.size());
            
            for (int idx : batch_indices) {
                try {
                    auto tensor = load_image(idx);
                    batch_data.push_back(std::move(tensor));
                    batch_labels.push_back(storage_backend->get_label_index(idx));
                } catch (const std::exception& e) {
                    std::cerr << "Error loading image " << idx << ": " << e.what() << std::endl;
                    continue;
                }
            }
            
            if (batch_data.empty()) continue;
            
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_not_full.wait(lock, [this]() {
                return prefetch_queue.size() < prefetch_factor || should_stop;
            });
            
            if (should_stop) break;
            
            PrefetchItem item{std::move(batch_data), std::move(batch_labels)};
            prefetch_queue.push(std::move(item));
            queue_not_empty.notify_one();
            
        } catch (const std::exception& e) {
            std::cerr << "Error in prefetch worker: " << e.what() << std::endl;
            break;
        }
    }
}

torch::Tensor PrefetchManager::load_image(int idx) {
    try {
        if (idx < 0 || idx >= storage_backend->get_num_samples()) {
            throw std::out_of_range("Invalid image index");
        }
        
        const Sample& sample = storage_backend->get_sample(idx);
        std::vector<unsigned char> buffer(sample.size);
        storage_backend->fetch(idx, buffer.data());
        
        // Decode image using OpenCV
        cv::Mat img = cv::imdecode(buffer, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Failed to decode image");
        }
        
        // Convert to RGB
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        
        // Convert to tensor
        auto tensor = torch::from_blob(rgb.data, {rgb.rows, rgb.cols, 3}, torch::kByte);
        tensor = tensor.permute({2, 0, 1}); // HWC to CHW
        tensor = tensor.to(torch::kFloat32).div(255.0); // Normalize to [0,1]
        
        return tensor;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error loading image: ") + e.what());
    }
}

PrefetchItem PrefetchManager::get_next_batch() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    queue_not_empty.wait(lock, [this]() {
        return !prefetch_queue.empty() || should_stop;
    });
    
    if (prefetch_queue.empty()) {
        return PrefetchItem{};  // Return empty batch
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
        PrefetchItem batch = get_next_batch();
        if (batch.data.empty()) {
            throw py::stop_iteration();
        }
        
        auto data_tensor = torch::stack(batch.data);
        auto labels_tensor = torch::tensor(batch.labels);
        
        return py::make_tuple(data_tensor, labels_tensor);
    } catch (const std::exception& e) {
        throw py::stop_iteration();
    }
}

void PrefetchManager::debug_info() const {
    std::cout << "PrefetchManager Debug Info:" << std::endl;
    std::cout << " - Queue size: " << prefetch_queue.size() << std::endl;
    std::cout << " - Prefetch factor: " << prefetch_factor << std::endl;
    std::cout << " - Number of workers: " << num_workers << std::endl;
    std::cout << " - Worker threads: " << worker_threads.size() << std::endl;
    std::cout << " - Should stop: " << (should_stop ? "true" : "false") << std::endl;
}