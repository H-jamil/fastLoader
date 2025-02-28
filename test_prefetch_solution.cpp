#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <thread>
#include <mpi.h>
#include <torch/torch.h>
#include "../include/distributed/DistributedManager.h"
#include "../include/storage/FileSystemBackend.h"
#include "../include/prefetcher/BufferManager.h"
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
    // Check command line arguments
    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <directory>" << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::string directory = argv[1];
    if (rank == 0) {
        std::cout << "=== Testing Prefetching Solution with " << world_size 
                  << " nodes on dataset: " << directory << " ===" << std::endl;
    }
    // ----------- Step 1: Initialize FileSystem Backend -----------
    auto start_time = std::chrono::high_resolution_clock::now();
    FileSystemBackend backend(directory);
    auto end_time = std::chrono::high_resolution_clock::now();
    double scan_time = std::chrono::duration<double>(end_time - start_time).count();
    int global_num_samples = backend.get_num_samples();
    std::cout << "Dataset scan time: " << scan_time << " seconds" << std::endl;
    std::cout << "Global number of samples: " << global_num_samples << std::endl;
    std::cout << "Number of classes: " << backend.get_num_classes() << std::endl;
    
    // ----------- Step 2: Initialize Distributed Manager -----------
    const int batch_size = 32;  // Global batch size
    DistributedManager dist_manager(global_num_samples, batch_size);

    // ----------- Step 3: Initialize BufferManager and PrefetchManager -----------
    // Buffer large enough to hold a few batches worth of data
    // Assuming average image size of 150KB, buffer can hold ~20 images
    const size_t buffer_size = 100 * 1024 * 1024;  // 3MB buffer
    BufferManager buffer_manager(buffer_size, batch_size);
    // PrefetchManager configuration
    const int prefetch_factor = 2;  // Number of batches to prefetch
    const int num_workers = 2;      // Number of worker threads
    PrefetchManager prefetch_manager(&backend, &dist_manager, prefetch_factor, num_workers, batch_size);
    // ----------- Step 4: Run tests for 3 epochs -----------
    const int num_epochs = 5;
    // Store partitions to check for uniqueness across epochs
    std::vector<std::set<int>> previous_epoch_partitions;

    for (int ep = 0; ep < num_epochs; ++ep) {
        if (rank == 0) {
            std::cout << "\n=== Starting Epoch " << ep+1 << " ===" << std::endl;
        }
        
        // Reset epoch and get partition
        dist_manager.reset_epoch();
        const std::vector<int>& partition = dist_manager.get_epoch_indices();
        int local_partition_size = partition.size();

        // Print node-specific information
        std::cout << "Node Rank: " << rank 
                  << ", Epoch: " << ep + 1 
                  << ", Partition size: " << local_partition_size << std::endl;

        // --- Condition 1: Check global mutual exclusivity & completeness ---
        // Gather partition sizes from all nodes
        std::vector<int> sizes(world_size, 0);
        MPI_Allgather(&local_partition_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        // Gather all partitioned indices
        int total_indices = 0;
        for (int s : sizes) total_indices += s;
        std::vector<int> all_indices(total_indices);
        std::vector<int> displs(world_size, 0);
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i-1] + sizes[i-1];
        }
        MPI_Allgatherv(partition.data(), local_partition_size, MPI_INT,
                       all_indices.data(), sizes.data(), displs.data(), MPI_INT,
                       MPI_COMM_WORLD);
        std::set<int> union_set(all_indices.begin(), all_indices.end());
        bool cond1 = (union_set.size() == static_cast<size_t>(global_num_samples));
        if (rank == 0) {
            if (cond1)
                std::cout << "Epoch " << ep+1 << ": Condition 1 PASSED (global dataset complete and mutually exclusive)" << std::endl;
            else
                std::cout << "Epoch " << ep+1 << ": Condition 1 FAILED" << std::endl;
        }
        // --- Condition 2: Check that partition for this epoch is different from previous epochs ---
        std::set<int> current_partition_set(partition.begin(), partition.end());
        bool cond2 = true;
        for (const auto& prev : previous_epoch_partitions) {
            if (prev == current_partition_set) { 
                cond2 = false;
                break;
            }
        }
        if (rank == 0) {
            if (cond2)
                std::cout << "Epoch " << ep+1 << ": Condition 2 PASSED (partition different from previous epochs)" << std::endl;
            else
                std::cout << "Epoch " << ep+1 << ": Condition 2 FAILED" << std::endl;
        }
        previous_epoch_partitions.push_back(current_partition_set);
        // ----------- Step 5: Start prefetching and process batches -----------
        prefetch_manager.start_prefetching();
        
        // --- Process mini-batches and check Condition 3: mini-batches should be disjoint ---
        std::vector<std::set<int>> mini_batches_collected;
        int batch_no = 1;
        int total_images_processed = 0;
        
        // Timing for prefetching performance
        auto prefetch_start = std::chrono::high_resolution_clock::now();
        
        // Process batches until the epoch is complete
        while (true) {
            try {
                // Try to get next batch from prefetch manager
                PrefetchItem batch = prefetch_manager.get_next_batch();
                
                if (batch.data.empty()) {
                    // If we get an empty batch, we're at the end of the epoch
                    break;
                }
                
                // Get the mini-batch from the distributed manager to check for disjointness
                std::vector<int> mini_batch = dist_manager.get_next_batch_indices();
                if (mini_batch.empty()) {
                    break;
                }
                
                int mini_batch_size = mini_batch.size();
                total_images_processed += mini_batch_size;
                
                // Print batch information
                std::cout << "Node Rank: " << rank 
                          << ", Epoch: " << ep + 1 
                          << ", Batch " << batch_no 
                          << ", Mini-batch size: " << mini_batch_size
                          << ", Tensors: " << batch.data.size() << std::endl;
                
                // Optional: Print information about the first tensor in the batch
                if (!batch.data.empty() && rank == 0 && batch_no == 1) {
                    print_tensor_info(batch.data[0], "First image in batch");
                }
                
                // Check for disjoint mini-batches (Condition 3)
                std::set<int> mini_batch_set(mini_batch.begin(), mini_batch.end());
                bool cond3 = true;
                for (const auto& prev_batch : mini_batches_collected) {
                    // Check for overlap with previous batches
                    std::vector<int> intersection;
                    std::set_intersection(prev_batch.begin(), prev_batch.end(),
                                        mini_batch_set.begin(), mini_batch_set.end(),
                                        std::back_inserter(intersection));
                    if (!intersection.empty()) {
                        cond3 = false;
                        break;
                    }
                }
                
                if (rank == 0) {
                    if (cond3)
                        std::cout << "Epoch " << ep+1 << ", Batch " << batch_no 
                                << ": Condition 3 PASSED (mini-batch distinct from previous ones)" << std::endl;
                    else
                        std::cout << "Epoch " << ep+1 << ", Batch " << batch_no 
                                << ": Condition 3 FAILED" << std::endl;
                }
                
                mini_batches_collected.push_back(mini_batch_set);
                ++batch_no;
                
            } catch (const std::exception& e) {
                std::cerr << "Error in batch processing at rank " << rank 
                          << ", epoch " << ep+1 << ": " << e.what() << std::endl;
                break;
            }
        }
        
        auto prefetch_end = std::chrono::high_resolution_clock::now();
        double prefetch_time = std::chrono::duration<double>(prefetch_end - prefetch_start).count();
        
        // Stop prefetching for this epoch
        prefetch_manager.stop_prefetching();
        
        // Print performance metrics
        std::cout << "Node Rank: " << rank 
                  << ", Epoch: " << ep + 1 
                  << ", Total images processed: " << total_images_processed
                  << ", Processing time: " << prefetch_time << " seconds"
                  << ", Images/second: " << total_images_processed / prefetch_time << std::endl;
        
        // Synchronize nodes before moving to the next epoch
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        std::cout << "\n=== Prefetching Solution Test Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}