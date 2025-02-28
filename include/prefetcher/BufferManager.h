#ifndef DIST_DATALOADER_BUFFER_MANAGER_H
#define DIST_DATALOADER_BUFFER_MANAGER_H

#include <mutex>
#include <condition_variable>
#include <queue>

class BufferManager {
public:
    BufferManager(size_t buffer_size, size_t batch_size);
    ~BufferManager();

    void write_to_buffer(const unsigned char* data, size_t size, int file_id);
    std::pair<unsigned char*, size_t> get_next_batch();
    bool is_full() const;
    bool is_empty() const;

private:
    unsigned char* buffer;
    size_t buffer_size;
    size_t batch_size;
    size_t write_pos;
    size_t read_pos;
    
    std::mutex mutex;
    std::condition_variable not_full;
    std::condition_variable not_empty;
    
    std::queue<std::pair<size_t, size_t>> batch_boundaries;
};

#endif