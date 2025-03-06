#ifndef DIST_DATALOADER_FILESYSTEM_BACKEND_H
#define DIST_DATALOADER_FILESYSTEM_BACKEND_H

#include "StorageBackend.h"
#include <unordered_map>
#include <vector>
#include <string>

class FileSystemBackend : public StorageBackend {
public:
    explicit FileSystemBackend(const std::string& data_path);
    ~FileSystemBackend() override = default;
    void fetch(int file_id, unsigned char* dst) override;
    size_t get_file_size(int file_id) const override;
    int get_num_samples() const override;
    std::string get_label(int file_id) const override;
    int get_label_index(int file_id) const override;
    const Sample& get_sample(int file_id) const override;
    int get_num_classes() const override;
    const std::vector<Sample>& get_samples() const override;

private:
    std::string data_path;
    std::vector<Sample> samples;
    std::vector<std::string> index_to_label;
    std::unordered_map<std::string, int> label_to_idx;
    void scan_directory();
};

#endif