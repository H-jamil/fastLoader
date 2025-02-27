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
    label_to_idx.clear();
    try {
        std::vector<std::string> class_names;
        // First pass: collect class names
        for (const auto& class_entry : fs::directory_iterator(data_path)) {
            if (class_entry.is_directory()) {
                class_names.push_back(class_entry.path().filename().string());
            }
        }

        // Sort class names for consistent indexing
        std::sort(class_names.begin(), class_names.end());

        // Create label mapping
        for (size_t i = 0; i < class_names.size(); i++) {
            label_to_idx[class_names[i]] = static_cast<int>(i);
        }

        // Second pass: collect samples
        for (const auto& class_entry : fs::directory_iterator(data_path)) {
            if (!class_entry.is_directory()) {
                continue;
            }
        
            std::string class_name = class_entry.path().filename().string();

            for (const auto& sample_entry : fs::directory_iterator(class_entry.path())) {
                if (!sample_entry.is_regular_file()) {
                    continue;
                }

                if (sample_entry.path().extension() != ".JPEG") {
                    continue;
                }
                
                Sample sample;
                sample.path = sample_entry.path().string();
                sample.size = fs::file_size(sample_entry.path());
                sample.label = class_name;

                samples.push_back(sample);
            }
        }

        // Sort samples by path for consistent ordering
        std::sort(samples.begin(), samples.end(),
                 [](const Sample& a, const Sample& b) {
                     return a.path < b.path;
                 });
        
        if (samples.empty()) {
            throw std::runtime_error("No valid samples found in directory: " + data_path);
        }
        
        std::cout << "Found " << samples.size() << " samples across " 
                  << label_to_idx.size() << " classes in " 
                  << data_path << std::endl;
        
    } catch (const fs::filesystem_error& e) {
        throw std::runtime_error("Error scanning directory: " + std::string(e.what()));
    }
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
    return samples[file_id].label;
}

int FileSystemBackend::get_label_index(int file_id) const {
    if (file_id < 0 || static_cast<size_t>(file_id) >= samples.size()) {
        throw std::out_of_range("Invalid file_id");
    }
    auto it = label_to_idx.find(samples[file_id].label);
    if (it == label_to_idx.end()) {
        throw std::runtime_error("Label not found in mapping: " + samples[file_id].label);
    }
    return it->second;
}

int FileSystemBackend::get_num_classes() const {
    return static_cast<int>(label_to_idx.size());
}

const Sample& FileSystemBackend::get_sample(int file_id) const {
    if (file_id < 0 || static_cast<size_t>(file_id) >= samples.size()) {
        throw std::out_of_range("Invalid file_id");
    }
    return samples[file_id];
}