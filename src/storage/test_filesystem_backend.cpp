   // src/test_filesystem_backend.cpp
   #include <iostream>
   #include <chrono>
   #include <filesystem>
   #include "../include/storage/FileSystemBackend.h"

   void test_filesystem_backend(const std::string& directory) {
       // Start timing the scan
       auto start = std::chrono::high_resolution_clock::now();

       // Create an instance of FileSystemBackend
       FileSystemBackend backend(directory);

       // Stop timing the scan
       auto end = std::chrono::high_resolution_clock::now();
       std::chrono::duration<double> scan_time = end - start;

       std::cout << "Scanning time: " << scan_time.count() << " seconds" << std::endl;
       std::cout << "Number of samples: " << backend.get_num_samples() << std::endl;
       std::cout << "Number of classes: " << backend.get_num_classes() << std::endl;

       // Example of fetching a sample
       if (backend.get_num_samples() > 0) {
           unsigned char* buffer = new unsigned char[backend.get_file_size(0)];
           backend.fetch(0, buffer);
           std::cout << "Fetched sample size: " << backend.get_file_size(0) << " bytes" << std::endl;
           delete[] buffer;
       }
   }

   int main(int argc, char* argv[]) {
       if (argc < 2) {
           std::cerr << "Usage: " << argv[0] << " <directory>" << std::endl;
           return 1;
       }

       std::string directory = argv[1];
       test_filesystem_backend(directory);
       return 0;
   }