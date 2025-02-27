#ifndef DIST_DATALOADER_STORAGE_BACKEND_H
#define DIST_DATALOADER_STORAGE_BACKEND_H

#include <string>
#include <vector>

struct Sample {
    std::string path;
    std::string label;
    size_t size;
};

class StorageBackend {
public:
    virtual ~StorageBackend() = default;
    virtual void fetch(int file_id, unsigned char* dst) = 0;
    virtual size_t get_file_size(int file_id) const = 0;
    virtual int get_num_samples() const = 0;
    virtual std::string get_label(int file_id) const = 0;
    virtual int get_label_index(int file_id) const = 0;
    virtual const Sample& get_sample(int file_id) const = 0;
    virtual int get_num_classes() const = 0;
};

#endif
