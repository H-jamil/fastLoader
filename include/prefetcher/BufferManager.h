#ifndef DIST_DATALOADER_BUFFER_MANAGER_H
#define DIST_DATALOADER_BUFFER_MANAGER_H

#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <memory>
#include <numeric>

struct BatchMetadata {
    size_t start_pos;
    size_t total_size;
    std::vector<size_t> sample_sizes;  // Size of each sample in the batch
    std::vector<int> sample_indices;   // Original indices of samples in the batch
};

class BufferManager {
public:
    BufferManager(size_t buffer_size, size_t batch_size);
    ~BufferManager();

    // Write a complete batch to the buffer
    void write_batch_to_buffer(const std::vector<unsigned char*>& batch_data,
                              const std::vector<size_t>& batch_sizes,
                              const std::vector<int>& batch_indices);
    
    // Get the next batch from the buffer
    std::pair<std::vector<unsigned char*>, BatchMetadata> get_next_batch();
    
    // Check buffer status
    bool is_full() const;
    bool is_empty() const;
    size_t get_available_space() const;
    size_t get_used_space() const;
    
    // Clear the buffer
    void clear();

private:
    unsigned char* buffer;
    size_t buffer_size;
    size_t batch_size;
    mutable size_t write_pos;
    mutable size_t read_pos;
    
    mutable std::mutex mutex;
    std::condition_variable not_full;
    std::condition_variable not_empty;
    
    std::queue<BatchMetadata> batch_metadata_queue;
    
    // Helper methods
    bool has_enough_space(const std::vector<size_t>& batch_sizes) const;
    void advance_write_pos(size_t amount);
    void advance_read_pos(size_t amount);
};

#endif