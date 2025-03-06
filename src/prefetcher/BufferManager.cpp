// src/prefetcher/BufferManager.cpp
#include "../../include/prefetcher/BufferManager.h"
#include <cstring>
#include <algorithm>

BufferManager::BufferManager(size_t buffer_size, size_t batch_size)
    : buffer_size(buffer_size), batch_size(batch_size), write_pos(0), read_pos(0) {
    buffer = new unsigned char[buffer_size];
}

BufferManager::~BufferManager() {
    delete[] buffer;
}

bool BufferManager::has_enough_space(const std::vector<size_t>& batch_sizes) const {
    size_t total_batch_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), size_t(0));
    return (write_pos + total_batch_size) <= buffer_size;
}

void BufferManager::advance_write_pos(size_t amount) {
    write_pos += amount;
    if (write_pos >= buffer_size) {
        write_pos = 0;  // Wrap around
    }
}

void BufferManager::advance_read_pos(size_t amount) {
    read_pos += amount;
    if (read_pos >= buffer_size) {
        read_pos = 0;  // Wrap around
    }
}

void BufferManager::write_batch_to_buffer(const std::vector<unsigned char*>& batch_data,
                                        const std::vector<size_t>& batch_sizes,
                                        const std::vector<int>& batch_indices) {
    std::unique_lock<std::mutex> lock(mutex);
    
    // Calculate total batch size
    size_t total_batch_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), size_t(0));
    
    // Wait until there's enough space
    not_full.wait(lock, [this, total_batch_size]() {
        return has_enough_space({total_batch_size});
    });
    
    // Create batch metadata
    BatchMetadata metadata;
    metadata.start_pos = write_pos;
    metadata.total_size = total_batch_size;
    metadata.sample_sizes = batch_sizes;
    metadata.sample_indices = batch_indices;
    
    // Write data to buffer
    size_t current_pos = write_pos;
    for (size_t i = 0; i < batch_data.size(); ++i) {
        std::memcpy(buffer + current_pos, batch_data[i], batch_sizes[i]);
        current_pos += batch_sizes[i];
        if (current_pos >= buffer_size) {
            current_pos = 0;  // Wrap around
        }
    }
    
    // Update write position and queue metadata
    advance_write_pos(total_batch_size);
    batch_metadata_queue.push(std::move(metadata));
    
    not_empty.notify_one();
}

std::pair<std::vector<unsigned char*>, BatchMetadata> BufferManager::get_next_batch() {
    std::unique_lock<std::mutex> lock(mutex);
    
    not_empty.wait(lock, [this]() {
        return !batch_metadata_queue.empty();
    });
    
    const BatchMetadata& metadata = batch_metadata_queue.front();
    std::vector<unsigned char*> batch_data;
    batch_data.reserve(metadata.sample_sizes.size());
    
    // Extract batch data
    size_t current_pos = metadata.start_pos;
    for (size_t size : metadata.sample_sizes) {
        unsigned char* sample_data = new unsigned char[size];
        std::memcpy(sample_data, buffer + current_pos, size);
        batch_data.push_back(sample_data);
        
        current_pos += size;
        if (current_pos >= buffer_size) {
            current_pos = 0;  // Wrap around
        }
    }
    
    // Update read position and remove metadata
    advance_read_pos(metadata.total_size);
    batch_metadata_queue.pop();
    
    not_full.notify_one();
    
    return {std::move(batch_data), metadata};
}

bool BufferManager::is_full() const {
    std::lock_guard<std::mutex> lock(mutex);
    return batch_metadata_queue.size() >= batch_size;
}

bool BufferManager::is_empty() const {
    std::lock_guard<std::mutex> lock(mutex);
    return batch_metadata_queue.empty();
}

size_t BufferManager::get_available_space() const {
    std::lock_guard<std::mutex> lock(mutex);
    return buffer_size - get_used_space();
}

size_t BufferManager::get_used_space() const {
    std::lock_guard<std::mutex> lock(mutex);
    size_t used = 0;
    // Create a copy of the queue to iterate over
    std::queue<BatchMetadata> queue_copy = batch_metadata_queue;
    while (!queue_copy.empty()) {
        used += queue_copy.front().total_size;
        queue_copy.pop();
    }
    return used;
}

void BufferManager::clear() {
    std::unique_lock<std::mutex> lock(mutex);
    write_pos = 0;
    read_pos = 0;
    std::queue<BatchMetadata>().swap(batch_metadata_queue);
    not_full.notify_all();
}