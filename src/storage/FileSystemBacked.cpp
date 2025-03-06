#include "../../include/storage/FileSystemBackend.h"
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

FileSystemBackend::FileSystemBackend(const std::string& data_path) 
    : data_path(data_path) {
    if (!fs::exists(data_path) || !fs::is_directory(data_path)) {
        throw std::runtime_error("Invalid dataset path: " + data_path);
    }
    scan_directory();
}

void FileSystemBackend::scan_directory() {
    samples.clear();
    index_to_label.clear();
    label_to_idx.clear();

    // First pass: collect class names
    std::vector<std::string> class_names;
    for (const auto& class_entry : fs::directory_iterator(data_path)) {
        if (class_entry.is_directory()) {
            class_names.push_back(class_entry.path().filename().string());
        }
    }

    // Sort class names for consistent indexing
    std::sort(class_names.begin(), class_names.end());

    // Create bidirectional mapping
    for (size_t i = 0; i < class_names.size(); i++) {
        label_to_idx[class_names[i]] = static_cast<int>(i);
        index_to_label.push_back(class_names[i]);  // Store string labels in order
    }

    // Second pass: collect samples with numeric indices
    for (const auto& class_entry : fs::directory_iterator(data_path)) {
        if (!class_entry.is_directory()) continue;
        
        std::string class_name = class_entry.path().filename().string();
        int label_index = label_to_idx[class_name];  // Get numeric index once
        
        for (const auto& sample_entry : fs::directory_iterator(class_entry.path())) {
            if (!sample_entry.is_regular_file()) continue;
            if (sample_entry.path().extension() != ".JPEG") continue;
            
            Sample sample;
            sample.path = sample_entry.path().string();
            sample.size = fs::file_size(sample_entry.path());
            sample.label_index = label_index;  // Store numeric index directly
            samples.push_back(sample);
        }
    }

    // Clear the temporary mapping as it's no longer needed
    label_to_idx.clear();

    // Sort samples by path for consistent ordering
    std::sort(samples.begin(), samples.end(),
             [](const Sample& a, const Sample& b) {
                 return a.path < b.path;
             });
    
    if (samples.empty()) {
        throw std::runtime_error("No valid samples found in directory: " + data_path);
    }
    
    std::cout << "Found " << samples.size() << " samples across " 
              << index_to_label.size() << " classes in " 
              << data_path << std::endl;
}

void FileSystemBackend::fetch(int file_id, unsigned char* dst) {
    if (file_id < 0 || static_cast<size_t>(file_id) >= samples.size()) {
        throw std::out_of_range("Invalid file_id");
    }
    
    std::ifstream file(samples[file_id].path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + samples[file_id].path);
    }

    file.read(reinterpret_cast<char*>(dst), samples[file_id].size);
    if (!file) {
        throw std::runtime_error("Failed to read file: " + samples[file_id].path);
    }
}

size_t FileSystemBackend::get_file_size(int file_id) const {
    if (file_id < 0 || static_cast<size_t>(file_id) >= samples.size()) {
        throw std::out_of_range("Invalid file_id");
    }
    return samples[file_id].size;
}

int FileSystemBackend::get_num_samples() const {
    return static_cast<int>(samples.size());
}

std::string FileSystemBackend::get_label(int file_id) const {
    if (file_id < 0 || static_cast<size_t>(file_id) >= samples.size()) {
        throw std::out_of_range("Invalid file_id");
    }
    // Get string label from index_to_label map
    return index_to_label.at(samples[file_id].label_index);
}

int FileSystemBackend::get_label_index(int file_id) const {
    if (file_id < 0 || static_cast<size_t>(file_id) >= samples.size()) {
        throw std::out_of_range("Invalid file_id");
    }
    // Simply return the stored index
    return samples[file_id].label_index;
}

int FileSystemBackend::get_num_classes() const {
    return static_cast<int>(index_to_label.size());
}

const Sample& FileSystemBackend::get_sample(int file_id) const {
    if (file_id < 0 || static_cast<size_t>(file_id) >= samples.size()) {
        throw std::out_of_range("Invalid file_id");
    }
    return samples[file_id];
}

const std::vector<Sample>& FileSystemBackend::get_samples() const {
    return samples;
}