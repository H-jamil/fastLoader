#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <algorithm>
#include <mpi.h>
#include "../include/distributed/DistributedManager.h"
#include "../include/storage/FileSystemBackend.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <directory>" << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::string directory = argv[1];

    // ----------- Filesystem Backend: Scan Remote Directory -----------
    auto start = std::chrono::high_resolution_clock::now();
    FileSystemBackend backend(directory);
    auto end = std::chrono::high_resolution_clock::now();
    double scan_time = std::chrono::duration<double>(end - start).count();
    int global_num_samples = backend.get_num_samples();

    if (rank == 0) {
        std::cout << "Scanning time: " << scan_time << " seconds" << std::endl;
        std::cout << "Global number of samples: " << global_num_samples << std::endl;
        std::cout << "Number of classes: " << backend.get_num_classes() << std::endl;
    }

    // Optionally, fetch a sample to verify access.
    if (global_num_samples > 0) {
        unsigned char* buffer = new unsigned char[backend.get_file_size(0)];
        backend.fetch(0, buffer);
        if (rank == 0)
            std::cout << "Fetched sample size: " << backend.get_file_size(0) << " bytes" << std::endl;
        delete[] buffer;
    }

    // ----------- Distributed Manager: Create Dataset Partitions -----------
    // Global mini-batch size (here used directly by DistributedManager).
    const int batch_size = 32;
    DistributedManager manager(global_num_samples, batch_size);

    // To check that the partitioned indices change across epochs, we store them locally.
    std::vector<std::set<int>> previous_epoch_partitions;

    const int num_epochs = 3;
    for (int ep = 0; ep < num_epochs; ++ep) {
        manager.reset_epoch();
        const std::vector<int>& partition = manager.get_epoch_indices();
        int local_partition_size = partition.size();

        // Print partitioned subset size for this node.
        std::cout << "Node Rank: " << manager.get_rank() 
                  << ", Epoch: " << ep + 1 
                  << ", Partition size: " << local_partition_size << std::endl;

        // --- Condition 1: Check global mutual exclusivity & completeness ---
        // Gather partition sizes from all nodes.
        std::vector<int> sizes(world_size, 0);
        MPI_Allgather(&local_partition_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        // Now, gather the actual partitioned indices from all nodes.
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

        // --- Process mini-batches and check Condition 3: mini-batches should be disjoint ---
        std::vector<std::set<int>> mini_batches_collected;
        int batch_no = 1;
        while (true) {
            std::vector<int> mini_batch = manager.get_next_batch_indices();
            if (mini_batch.empty())
                break;
            int mini_batch_size = mini_batch.size();
            std::cout << "Node Rank: " << manager.get_rank() 
                      << ", Epoch: " << ep + 1 
                      << ", Batch " << batch_no 
                      << ", Mini-batch size: " << mini_batch_size << std::endl;
            std::set<int> mini_batch_set(mini_batch.begin(), mini_batch.end());
            bool cond3 = true;
            for (const auto& prev_batch : mini_batches_collected) {
                // Check if there is any overlap.
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
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
