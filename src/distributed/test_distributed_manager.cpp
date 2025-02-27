#include <iostream>
#include <mpi.h>
#include "../include/distributed/DistributedManager.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    // Define total number of samples and global batch size.
    const int num_samples = 10; 
    const int batch_size = 2;     

    // Create an instance of DistributedManager.
    DistributedManager manager(num_samples, batch_size);

    const int num_epochs = 5; // Simulate 5 epochs.
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Reset epoch (this shuffles and partitions indices).
        manager.reset_epoch();

        // Print the partitioned indices for this node in the current epoch.
        const std::vector<int>& partitioned = manager.get_epoch_indices();
        std::cout << "Node Rank: " << manager.get_rank() 
                  << ", Epoch: " << epoch + 1 
                  << ", Partitioned Indices: ";
        for (int idx : partitioned) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;

        // Print all batches for this epoch.
        int batch_no = 1;
        while (true) {
            std::vector<int> batch = manager.get_next_batch_indices();
            if (batch.empty()) {
                break; // End of epoch.
            }
            std::cout << "Node Rank: " << manager.get_rank() 
                      << ", Epoch: " << epoch + 1 
                      << ", Batch " << batch_no << ": ";
            for (int idx : batch) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
            ++batch_no;
        }

        // Synchronize nodes before moving to the next epoch.
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
