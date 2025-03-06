   // src/test_filesystem_backend.cpp
   #include <iostream>
   #include <chrono>
   #include <filesystem>
   #include <iomanip>
   #include "../include/storage/FileSystemBackend.h"

   void print_performance_metrics(const std::string& operation, double time) {
       std::cout << std::fixed << std::setprecision(6) 
                 << operation << ": " << time << " seconds" << std::endl;
   }

   void test_filesystem_backend(const std::string& directory) {
       std::cout << "\n=== FileSystemBackend Performance Test ===\n" << std::endl;

       // Test 1: Directory Scanning
       auto scan_start = std::chrono::high_resolution_clock::now();
       FileSystemBackend backend(directory);
       auto scan_end = std::chrono::high_resolution_clock::now();
       print_performance_metrics("Directory scanning time", 
           std::chrono::duration<double>(scan_end - scan_start).count());

       // Print dataset statistics
       std::cout << "\nDataset Statistics:" << std::endl;
       std::cout << "Total samples: " << backend.get_num_samples() << std::endl;
       std::cout << "Number of classes: " << backend.get_num_classes() << std::endl;

       // Test 2: Label Access Performance
       if (backend.get_num_samples() > 0) {
           std::cout << "\nLabel Access Performance Test:" << std::endl;
           
           // Test string label access
           auto label_start = std::chrono::high_resolution_clock::now();
           for (int i = 0; i < 1000; i++) {
               backend.get_label(i % backend.get_num_samples());
           }
           auto label_end = std::chrono::high_resolution_clock::now();
           print_performance_metrics("String label access (1000 iterations)", 
               std::chrono::duration<double>(label_end - label_start).count());

           // Test numeric label access
           auto idx_start = std::chrono::high_resolution_clock::now();
           for (int i = 0; i < 1000; i++) {
               backend.get_label_index(i % backend.get_num_samples());
           }
           auto idx_end = std::chrono::high_resolution_clock::now();
           print_performance_metrics("Numeric label access (1000 iterations)", 
               std::chrono::duration<double>(idx_end - idx_start).count());
       }

       // Test 3: Sample Data Access
       if (backend.get_num_samples() > 0) {
           std::cout << "\nSample Data Access Test:" << std::endl;
           
           // Get first sample size
           size_t sample_size = backend.get_file_size(0);
           unsigned char* buffer = new unsigned char[sample_size];
           
           // Test data loading
           auto load_start = std::chrono::high_resolution_clock::now();
           backend.fetch(0, buffer);
           auto load_end = std::chrono::high_resolution_clock::now();
           print_performance_metrics("Sample data loading time", 
               std::chrono::duration<double>(load_end - load_start).count());
           
           std::cout << "Sample size: " << sample_size << " bytes" << std::endl;
           
           // Print sample metadata
           const auto& sample = backend.get_sample(0);
           std::cout << "\nSample Metadata:" << std::endl;
           std::cout << "Path: " << sample.path << std::endl;
           std::cout << "Label: " << backend.get_label(0) << std::endl;
           std::cout << "Label Index: " << sample.label_index << std::endl;
           
           delete[] buffer;
       }

       std::cout << "\n=== Test Complete ===\n" << std::endl;
   }

   int main(int argc, char* argv[]) {
       if (argc < 2) {
           std::cerr << "Usage: " << argv[0] << " <directory>" << std::endl;
           std::cerr << "Example: " << argv[0] << " /path/to/image/dataset" << std::endl;
           return 1;
       }

       std::string directory = argv[1];
       test_filesystem_backend(directory);
       return 0;
   }