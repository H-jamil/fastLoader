#include "../include/distributed/DistributedManager.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

DistributedManager::DistributedManager(int num_samples, int batch_size)
    : num_samples(num_samples),
      batch_size(batch_size),
      current_index(0),
      epoch(0)
{
    // Initialize MPI if not already initialized.
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            throw std::runtime_error("MPI implementation does not support MPI_THREAD_MULTIPLE");
        }
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Initialize RNG with a rank-specific seed.
    rng.seed(42 + rank);

    // Set up the first epoch.
    reset_epoch();
}

DistributedManager::~DistributedManager() {
    // Do not finalize MPI here.
}

void DistributedManager::shuffle_indices() {
    // Create a global vector of indices.
    std::vector<int> all_indices(num_samples);
    if (is_main_process()) {
        // Fill indices using 0-indexing to match typical array indexing (0 to num_samples-1)
        for (int i = 0; i < num_samples; i++) {
            all_indices[i] = i;
        }
        std::shuffle(all_indices.begin(), all_indices.end(), rng);
    }
    // Broadcast the entire shuffled vector to all processes.
    MPI_Bcast(all_indices.data(), num_samples, MPI_INT, 0, MPI_COMM_WORLD);

    // Partition the indices among processes.
    int local_sample_count = num_samples / world_size;
    int remainder = num_samples % world_size;
    int start = rank * local_sample_count + std::min(rank, remainder);
    int count = local_sample_count + (rank < remainder ? 1 : 0);

    // Clear any existing indices first
    epoch_indices.clear();
    // Assign the indices for this process
    epoch_indices.assign(all_indices.begin() + start, all_indices.begin() + start + count);
    current_index = 0;
}

std::vector<int> DistributedManager::get_next_batch_indices() {
    std::vector<int> batch_indices;
    int local_batch_size = get_local_batch_size();

    // Check if we have reached the end of the epoch
    if (current_index >= epoch_indices.size()) {
        return batch_indices; // Return an empty vector to indicate the end of the epoch
    }

    // Compute the end position for the current batch.
    int end = std::min(current_index + local_batch_size, static_cast<int>(epoch_indices.size()));
    
    // Insert the indices into the batch
    batch_indices.insert(batch_indices.end(),
                         epoch_indices.begin() + current_index,
                         epoch_indices.begin() + end);
    
    // If this is the last batch and it's smaller than local_batch_size,
    // pad it with the last element
    if (end == epoch_indices.size() && batch_indices.size() < local_batch_size) {
        int last_element = batch_indices.back();  // Get the last element
        while (batch_indices.size() < local_batch_size) {
            batch_indices.push_back(last_element);
        }
    }
    
    current_index = end; // Update current_index to the end of the batch
    return batch_indices; // Return the batch indices
}

void DistributedManager::reset_epoch() {
    epoch++;
    shuffle_indices();
}

int DistributedManager::get_rank() const {
    return rank;
}

int DistributedManager::get_world_size() const {
    return world_size;
}

int DistributedManager::get_local_batch_size() const {
    // Return the full global batch size for each node
    // (don't divide by world_size as we want each node to process batches of the same size)
    return batch_size;
}

bool DistributedManager::is_main_process() const {
    return rank == 0;
}
