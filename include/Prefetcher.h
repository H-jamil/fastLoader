#pragma once
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "FileSystemBackend.h"

struct Batch {
    std::vector<std::vector<char>> data;
    std::vector<int> labels;
};

class Prefetcher {
public:
    Prefetcher(const std::string& dataset_path, int batch_size, int num_threads);
    ~Prefetcher();
    
    Batch getNextBatch();
    void reset();
    bool hasNext();  // Removed const qualifier

private:
    void prefetchThread();
    void loadFileList();
    
    std::string dataset_path_;
    int batch_size_;
    int num_threads_;
    
    std::unique_ptr<FileSystemBackend> backend_;
    std::vector<std::string> file_paths_;
    std::vector<int> labels_;
    
    size_t current_index_;
    
    std::queue<Batch> batch_queue_;
    mutable std::mutex queue_mutex_;  // Made mutable to allow locking in const methods
    std::condition_variable queue_cv_;
    bool stop_threads_;
    std::vector<std::thread> prefetch_threads_;
};