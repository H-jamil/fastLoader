#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <mpi.h>
#include <torch/torch.h>
#include "../include/distributed/DistributedManager.h"
#include "../include/storage/FileSystemBackend.h"
#include "../include/prefetcher/PrefetchManager.h"

// Function to print tensor info for debugging
void print_tensor_info(const torch::Tensor& tensor, const std::string& name) {
    std::cout << name << ": shape=" << tensor.sizes() << ", dtype=" << tensor.dtype() 
              << ", device=" << tensor.device() << std::endl;
}

int main(int argc, char* argv[]) {
    // Initialize MPI with thread support for prefetching
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Warning: MPI implementation does not fully support MPI_THREAD_MULTIPLE" << std::endl;
    }
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 MPI processes" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Initialize FileSystem Backend with CephFS path
    const std::string cephfs_path = "/mnt/cephfs/subset/train/";
    auto start_time = std::chrono::high_resolution_clock::now();
    FileSystemBackend backend(cephfs_path);
    auto end_time = std::chrono::high_resolution_clock::now();
    double scan_time = std::chrono::duration<double>(end_time - start_time).count();
    
    int global_num_samples = backend.get_num_samples();
    if (rank == 0) {
        std::cout << "=== Testing Prefetching with CephFS ===" << std::endl;
        std::cout << "Dataset scan time: " << scan_time << " seconds" << std::endl;
        std::cout << "Global number of samples: " << global_num_samples << std::endl;
        std::cout << "Number of classes: " << backend.get_num_classes() << std::endl;
    }

    // Initialize Distributed Manager and PrefetchManager
    const int batch_size = 32;  // Global batch size
    DistributedManager dist_manager(global_num_samples, batch_size);
    
    // PrefetchManager configuration
    const size_t prefetch_factor = 8;  // Number of batches to prefetch
    const int initial_num_workers = 8;  // Initial number of worker threads
    PrefetchManager prefetch_manager(&backend, &dist_manager, prefetch_factor, initial_num_workers, batch_size);

    // Run multiple epochs with performance measurements
    const int num_epochs = 5;
    for (int ep = 0; ep < num_epochs; ++ep) {
        if (rank == 0) {
            std::cout << "\n=== Starting Epoch " << ep+1 << " ===" << std::endl;
        }
        
        // Reset epoch and get partition
        dist_manager.reset_epoch();
        const std::vector<int>& partition = dist_manager.get_epoch_indices();
        
        if (rank == 0) {
            std::cout << "Node " << rank << " partition size: " << partition.size() << std::endl;
        }
        
        // Start prefetching
        prefetch_manager.start_prefetching();
        
        // Process batches and measure performance
        auto epoch_start = std::chrono::high_resolution_clock::now();
        int total_batches = 0;
        int total_images = 0;
        
        // Optional: Adjust number of workers based on epoch
        if (ep == 2) {  // Example: increase workers in epoch 3
            prefetch_manager.set_num_workers(12);
            if (rank == 0) {
                std::cout << "Increased number of workers to 12" << std::endl;
            }
        }
        
        while (true) {
            try {
                // Get next batch
                PrefetchItem batch = prefetch_manager.get_next_batch();
                
                if (batch.data.empty()) {
                    // End of epoch
                    break;
                }
                
                total_batches++;
                total_images += batch.data.size();
                
                // Print progress every 10 batches
                if (total_batches % 10 == 0 && rank == 0) {
                    std::cout << "Processed " << total_batches << " batches, "
                              << total_images << " images" << std::endl;
                }
                
                // Optional: Print first batch info
                if (total_batches == 1 && rank == 0) {
                    print_tensor_info(batch.data[0], "First image in batch");
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing batch at rank " << rank 
                          << ": " << e.what() << std::endl;
                break;
            }
        }
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
        
        // Stop prefetching for this epoch
        prefetch_manager.stop_prefetching();
        
        // Print performance metrics
        std::cout << "Node " << rank << " Performance:" << std::endl;
        std::cout << "  - Total batches: " << total_batches << std::endl;
        std::cout << "  - Total images: " << total_images << std::endl;
        std::cout << "  - Epoch time: " << epoch_time << " seconds" << std::endl;
        std::cout << "  - Images/second: " << total_images / epoch_time << std::endl;
        
        // Synchronize nodes before next epoch
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        std::cout << "\n=== Prefetching Test Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
} 