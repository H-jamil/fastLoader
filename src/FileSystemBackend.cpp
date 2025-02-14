#include "FileSystemBackend.h"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <iostream>

std::vector<char> FileSystemBackend::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read file content
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Error reading file: " + path);
    }

    return buffer;
}

size_t FileSystemBackend::getFileSize(const std::string& path) {
    return std::filesystem::file_size(path);
}

bool FileSystemBackend::fileExists(const std::string& path) {
    return std::filesystem::exists(path);
}

void FileSystemBackend::loadFileList(const std::string& dataset_path) {
    dataset_path_ = dataset_path;
    file_paths_.clear();
    labels_.clear();
    
    // Map to store class to label mapping
    std::map<std::string, int> class_to_label;
    int current_label = 0;
    
    // Iterate through the ImageNet directory structure
    for (const auto& class_dir : std::filesystem::directory_iterator(dataset_path_)) {
        if (class_dir.is_directory()) {
            std::string class_name = class_dir.path().filename().string();
            
            // Assign label to class if not already assigned
            if (class_to_label.find(class_name) == class_to_label.end()) {
                class_to_label[class_name] = current_label++;
            }
            
            // Iterate through all images in class directory
            for (const auto& img_file : std::filesystem::directory_iterator(class_dir)) {
                if (img_file.is_regular_file()) {
                    file_paths_.push_back(img_file.path().string());
                    labels_.push_back(class_to_label[class_name]);
                }
            }
        }
    }
}