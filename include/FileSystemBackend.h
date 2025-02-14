#pragma once
#include "StorageBackend.h"
#include <string>
#include <vector>
#include <map>

class FileSystemBackend : public StorageBackend {
public:
    FileSystemBackend() = default;
    ~FileSystemBackend() = default;

    std::vector<char> readFile(const std::string& path) override;
    size_t getFileSize(const std::string& path) override;
    bool fileExists(const std::string& path) override;
    
    // Add these new methods
    void loadFileList(const std::string& dataset_path) override;
    const std::vector<std::string>& getFilePaths() const override { return file_paths_; }
    const std::vector<int>& getLabels() const override { return labels_; }

private:
    std::vector<std::string> file_paths_;
    std::vector<int> labels_;
    std::string dataset_path_;
};