#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <mpi.h>
#include "../include/prefetcher/PrefetchManager.h"
#include "../include/storage/FileSystemBackend.h"
#include "../include/distributed/DistributedManager.h"

namespace py = pybind11;

// This function will help us verify if MPI has been initialized and initialize it if needed
bool ensure_mpi_initialized() {
    int initialized;
    MPI_Initialized(&initialized);
    
    if (!initialized) {
        // Initialize MPI with thread support for prefetching
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            PyErr_WarnEx(PyExc_RuntimeWarning, 
                         "MPI implementation does not fully support MPI_THREAD_MULTIPLE", 1);
        }
        return true;
    }
    return false;
}

// Wrapper class that encapsulates our C++ prefetching functionality
class PyTorchPrefetcher {
private:
    FileSystemBackend* backend;
    DistributedManager* dist_manager;
    PrefetchManager* prefetch_manager;
    int batch_size;
    bool initialized;
    bool we_initialized_mpi;
    int rank;
    int world_size;
    bool preprocessing_enabled;
    std::pair<int, int> target_size;

public:
    PyTorchPrefetcher() : 
        backend(nullptr), 
        dist_manager(nullptr), 
        prefetch_manager(nullptr), 
        batch_size(32), 
        initialized(false), 
        we_initialized_mpi(false),
        rank(-1),
        world_size(0),
        preprocessing_enabled(false),
        target_size({224, 224}) {}
    
    void initialize(const std::string& data_path, int batch_size, 
                    int num_workers, int prefetch_factor,
                    bool preprocess = false, const std::pair<int, int>& preprocess_size = {224, 224}) {
        this->batch_size = batch_size;
        this->preprocessing_enabled = preprocess;
        this->target_size = preprocess_size;
        
        // Ensure MPI is initialized
        we_initialized_mpi = ensure_mpi_initialized();
        
        // Get MPI rank and world size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        
        // Initialize backend with data path
        backend = new FileSystemBackend(data_path);
        
        // Get global sample count and initialize distributed manager
        int global_num_samples = backend->get_num_samples();
        dist_manager = new DistributedManager(global_num_samples, batch_size);
        
        // Initialize prefetch manager with preprocessing options
        prefetch_manager = new PrefetchManager(
            backend, dist_manager, prefetch_factor, num_workers, batch_size,
            preprocessing_enabled, target_size);
        
        initialized = true;
    }
    
    py::tuple get_next_batch() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        
        // Get the next batch from the prefetch manager
        PrefetchItem batch = prefetch_manager->get_next_batch();
        
        // If batch is empty, return None to indicate end of epoch
        if (batch.data.empty()) {
            return py::make_tuple(py::none(), py::none());
        }
        
        // Instead of stacking, create Python lists of tensors
        py::list data_tensors;
        py::list labels;
        
        for (size_t i = 0; i < batch.data.size(); i++) {
            data_tensors.append(batch.data[i]);
            if (i < batch.labels.size()) {  // Ensure we have a label
                labels.append(batch.labels[i]);
            } else {
                labels.append(-1);  // Default label if missing
            }
        }
        
        return py::make_tuple(data_tensors, labels);
    }
    
    void set_num_workers(int num) {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        prefetch_manager->set_num_workers(num);
    }
    
    void start_prefetching() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        prefetch_manager->start_prefetching();
    }
    
    void stop_prefetching() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        prefetch_manager->stop_prefetching();
    }
    
    void reset_epoch() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        dist_manager->reset_epoch();
    }
    
    int get_rank() const {
        return rank;
    }
    
    int get_world_size() const {
        return world_size;
    }
    
    py::dict get_dataset_info() {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        
        py::dict info;
        info["num_samples"] = backend->get_num_samples();
        info["num_classes"] = backend->get_num_classes();
        return info;
    }
    
    // Debug info for troubleshooting
    void debug_info() {
        if (!initialized) {
            py::print("Prefetcher not initialized");
            return;
        }
        
        prefetch_manager->debug_info();
    }
    
    // Add a method to configure preprocessing at runtime
    void set_preprocessing(bool enable, const std::pair<int, int>& size = {224, 224}) {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        preprocessing_enabled = enable;
        target_size = size;
        prefetch_manager->set_preprocessing(enable, size);
    }
    
    bool is_preprocessing_enabled() const {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        return prefetch_manager->is_preprocessing_enabled();
    }
    
    std::pair<int, int> get_target_size() const {
        if (!initialized) throw std::runtime_error("Prefetcher not initialized");
        return prefetch_manager->get_target_size();
    }
    
    ~PyTorchPrefetcher() {
        if (prefetch_manager) {
            prefetch_manager->stop_prefetching();
            delete prefetch_manager;
        }
        if (dist_manager) delete dist_manager;
        if (backend) delete backend;
        
        // Finalize MPI if we initialized it
        if (we_initialized_mpi) {
            int finalized;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Finalize();
            }
        }
    }
};

PYBIND11_MODULE(fastloader, m) {
    m.doc() = "FastLoader: High-performance data loading with adaptive prefetching";
    
    py::class_<PyTorchPrefetcher>(m, "Prefetcher")
        .def(py::init<>())
        .def("initialize", &PyTorchPrefetcher::initialize, 
             py::arg("data_path"), py::arg("batch_size")=32, 
             py::arg("num_workers")=4, py::arg("prefetch_factor")=2,
             py::arg("preprocess")=false, py::arg("preprocess_size")=std::pair<int, int>{224, 224},
             "Initialize the prefetcher with dataset path and parameters, optionally enabling preprocessing")
        .def("get_next_batch", &PyTorchPrefetcher::get_next_batch,
             "Get the next batch of data, returns (data, labels) tuple")
        .def("set_num_workers", &PyTorchPrefetcher::set_num_workers,
             py::arg("num"),
             "Dynamically adjust the number of worker threads")
        .def("start_prefetching", &PyTorchPrefetcher::start_prefetching,
             "Start the prefetching process")
        .def("stop_prefetching", &PyTorchPrefetcher::stop_prefetching,
             "Stop the prefetching process")
        .def("reset_epoch", &PyTorchPrefetcher::reset_epoch,
             "Reset for a new epoch")
        .def("set_preprocessing", &PyTorchPrefetcher::set_preprocessing,
             py::arg("enable"), py::arg("size")=std::pair<int, int>{224, 224},
             "Enable or disable preprocessing and set target size")
        .def("is_preprocessing_enabled", &PyTorchPrefetcher::is_preprocessing_enabled,
             "Check if preprocessing is enabled")
        .def("get_target_size", &PyTorchPrefetcher::get_target_size,
             "Get the current preprocessing target size")
        .def("get_rank", &PyTorchPrefetcher::get_rank,
             "Get the MPI rank of this process")
        .def("get_world_size", &PyTorchPrefetcher::get_world_size,
             "Get the total number of MPI processes")
        .def("get_dataset_info", &PyTorchPrefetcher::get_dataset_info,
             "Get information about the dataset")
        .def("debug_info", &PyTorchPrefetcher::debug_info,
             "Print debug information");
} 