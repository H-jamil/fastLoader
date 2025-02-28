// src/prefetcher/BufferManager.cpp
#include "../../include/prefetcher/BufferManager.h"
#include <cstring>

BufferManager::BufferManager(size_t buffer_size, size_t batch_size)
    : buffer_size(buffer_size), batch_size(batch_size), write_pos(0), read_pos(0) {
    buffer = new unsigned char[buffer_size];
}

BufferManager::~BufferManager() {
    delete[] buffer;
}

void BufferManager::write_to_buffer(const unsigned char* data, size_t size, int file_id) {
    std::unique_lock<std::mutex> lock(mutex);
    
    // Wait until there's enough space
    not_full.wait(lock, [this, size]() {
        return write_pos + size <= buffer_size;
    });
    
    // Copy data to buffer
    std::memcpy(buffer + write_pos, data, size);
    batch_boundaries.push({write_pos, size});
    write_pos += size;
    
    not_empty.notify_one();
}

std::pair<unsigned char*, size_t> BufferManager::get_next_batch() {
    std::unique_lock<std::mutex> lock(mutex);
    
    not_empty.wait(lock, [this]() {
        return !batch_boundaries.empty();
    });
    
    auto [pos, size] = batch_boundaries.front();
    batch_boundaries.pop();
    
    return {buffer + pos, size};
}