#include "Prefetcher.h"
#include <filesystem>
#include <algorithm>

Prefetcher::Prefetcher(const std::string& dataset_path, int batch_size, int num_threads)
    : dataset_path_(dataset_path)
    , batch_size_(batch_size)
    , num_threads_(num_threads)
    , current_index_(0)
    , stop_threads_(false) {
    
    backend_ = std::make_unique<FileSystemBackend>();
    backend_->loadFileList(dataset_path);  // Load the file list
    
    // Get the file paths and labels
    file_paths_ = backend_->getFilePaths();
    labels_ = backend_->getLabels();
    
    // Start prefetch threads
    for (int i = 0; i < num_threads_; ++i) {
        prefetch_threads_.emplace_back(&Prefetcher::prefetchThread, this);
    }
}

Prefetcher::~Prefetcher() {
    stop_threads_ = true;
    queue_cv_.notify_all();
    for (auto& thread : prefetch_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void Prefetcher::loadFileList() {
    for (const auto& entry : std::filesystem::directory_iterator(dataset_path_)) {
        if (entry.is_regular_file()) {
            file_paths_.push_back(entry.path().string());
            // Simple label extraction from filename
            labels_.push_back(std::stoi(entry.path().stem().string()));
        }
    }
}

void Prefetcher::prefetchThread() {
    while (!stop_threads_) {
        size_t local_index;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (current_index_ >= file_paths_.size()) {
                continue;
            }
            local_index = current_index_;
            current_index_ += batch_size_;
        }

        Batch batch;
        for (size_t i = 0; i < batch_size_ && local_index + i < file_paths_.size(); ++i) {
            const auto& path = file_paths_[local_index + i];
            batch.data.push_back(backend_->readFile(path));
            batch.labels.push_back(labels_[local_index + i]);
        }

        if (!batch.data.empty()) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            batch_queue_.push(std::move(batch));
            queue_cv_.notify_one();
        }
    }
}

Batch Prefetcher::getNextBatch() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { 
        return !batch_queue_.empty() || current_index_ >= file_paths_.size(); 
    });
    
    if (batch_queue_.empty() && current_index_ >= file_paths_.size()) {
        throw std::runtime_error("No more batches available");
    }
    
    Batch batch = std::move(batch_queue_.front());
    batch_queue_.pop();
    return batch;
}

void Prefetcher::reset() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    current_index_ = 0;
    while (!batch_queue_.empty()) {
        batch_queue_.pop();
    }
}

bool Prefetcher::hasNext() {  // Removed const qualifier
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return current_index_ < file_paths_.size() || !batch_queue_.empty();
}