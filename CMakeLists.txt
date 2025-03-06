cmake_minimum_required(VERSION 3.10)
project(FastLoader)

# Set C++17 as the default standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

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

# Add the prefetching solution test
add_executable(test_prefetch_solution test_prefetch_solution.cpp)
target_include_directories(test_prefetch_solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_prefetch_solution PRIVATE 
    ${MPI_LIBRARIES} 
    ${TORCH_LIBRARIES} 
    ${OpenCV_LIBS} 
    Python3::Python
    prefetcher_lib
    storage_lib
    distributed_lib)

# For PyTorch, we need to set CXX flags as recommended by PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Make sure we install the binaries to a known location
install(TARGETS test_prefetch_solution
        RUNTIME DESTINATION bin)

# Add subdirectories
add_subdirectory(src/storage)
add_subdirectory(src/distributed)
add_subdirectory(src/prefetcher)

# Add test_prefetch_cephfs executable
add_executable(test_prefetch_cephfs test_prefetch_cephfs.cpp)
target_include_directories(test_prefetch_cephfs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test_prefetch_cephfs
    PRIVATE
    ${MPI_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Python3::Python
    prefetcher_lib
    storage_lib
    distributed_lib)

# Make sure we install the binaries to a known location
install(TARGETS test_prefetch_cephfs
        RUNTIME DESTINATION bin)

# Add test_prefetch_cephfs to the test targets
add_test(NAME test_prefetch_cephfs COMMAND test_prefetch_cephfs)