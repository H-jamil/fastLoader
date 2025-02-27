#ifndef DIST_DATALOADER_DISTRIBUTED_MANAGER_H
#define DIST_DATALOADER_DISTRIBUTED_MANAGER_H

#include <vector>
#include <random>
#include <mpi.h>

class DistributedManager {
public:
    DistributedManager(int num_samples, int batch_size);
    ~DistributedManager();

    // Returns the next batch of local indices
    std::vector<int> get_next_batch_indices();

    // Resets the epoch by shuffling and partitioning indices
    void reset_epoch();

    int get_rank() const;
    int get_world_size() const;
    int get_local_batch_size() const;
    bool is_main_process() const;
    const std::vector<int>& get_epoch_indices() const { return epoch_indices; }


private:
    int rank;
    int world_size;
    int num_samples;
    int batch_size;
    int current_index;
    int epoch;
    std::vector<int> epoch_indices; // Local partition for the current epoch
    std::mt19937 rng;
    
    // Shuffles the global indices (on main process) and broadcasts them to partition locally.
    void shuffle_indices();
};

#endif
