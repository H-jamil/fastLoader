cmake_minimum_required(VERSION 3.10)
project(DistDataLoader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get Python include paths from the active virtual environment
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python library path
execute_process(
    COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find required packages
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories - add Python include paths
include_directories(include ${MPI_INCLUDE_PATH} ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

# Link directories - add Python library path
link_directories(${PYTHON_LIBRARY_DIR})

# Source files
set(SOURCES
    src/storage/FileSystemBacked.cpp
    src/storage/test_filesystem_backend.cpp
)

# Create the executable
add_executable(test_filesystem_backend ${SOURCES})

# Add the prefetching solution test
add_executable(test_prefetch_solution test_prefetch_solution.cpp 
                                    src/distributed/DistributedManager.cpp
                                    src/storage/FileSystemBacked.cpp
                                    src/prefetcher/BufferManager.cpp
                                    src/prefetcher/PrefetchManager.cpp)
target_include_directories(test_prefetch_solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_prefetch_solution PRIVATE ${MPI_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS} python3.10)

# For PyTorch, we need to set CXX flags as recommended by PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Make sure we install the binaries to a known location
install(TARGETS test_prefetch_solution
        RUNTIME DESTINATION bin)