#pragma once
#include <string>
#include <vector>
#include <map>

class StorageBackend {
public:
    virtual ~StorageBackend() = default;
    virtual std::vector<char> readFile(const std::string& path) = 0;
    virtual size_t getFileSize(const std::string& path) = 0;
    virtual bool fileExists(const std::string& path) = 0;
    
    // Add these new methods
    virtual void loadFileList(const std::string& dataset_path) = 0;
    virtual const std::vector<std::string>& getFilePaths() const = 0;
    virtual const std::vector<int>& getLabels() const = 0;
};