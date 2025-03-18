cmake_minimum_required(VERSION 3.10)
project(FastLoader)

# Set C++17 as the default standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position independent code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get Python include paths from the active virtual environment
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get Python library path
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Automatically detect PyTorch path
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Set CMAKE_PREFIX_PATH to include the PyTorch path
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
message(STATUS "Using PyTorch from: ${TORCH_PATH}")

# Find PyTorch include directory - this is important for proper tensor handling
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(';'.join(torch.__config__.include_dirs))"
    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS_LIST
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Handle RPATH issues for Linux
if(UNIX AND NOT APPLE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Find or download pybind11
option(USE_SYSTEM_PYBIND11 "Use system pybind11 installation" OFF)
if(USE_SYSTEM_PYBIND11)
    find_package(pybind11 REQUIRED)
else()
    # If pybind11 is not found, we'll download and build it
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find required packages
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories - add Python include paths
include_directories(include ${MPI_INCLUDE_PATH} ${TORCH_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

# Link directories - add Python library path
link_directories(${PYTHON_LIBRARY_DIR})

# For PyTorch, we need to set CXX flags as recommended by PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add subdirectories
add_subdirectory(src/storage)
add_subdirectory(src/distributed)
add_subdirectory(src/prefetcher)

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

# Make sure we install the binaries to a known location
install(TARGETS test_prefetch_solution
        RUNTIME DESTINATION bin)

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

# Add the Python extension module
pybind11_add_module(fastloader src/prefetcher_python.cpp)

# Add include directories specifically for the Python module
target_include_directories(fastloader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

# This is crucial - link against torch_python library
target_link_libraries(fastloader PRIVATE 
    prefetcher_lib
    storage_lib
    distributed_lib
    ${MPI_LIBRARIES}
    "${TORCH_LIBRARIES}"
    ${OpenCV_LIBS}
    # This line is critical for PyTorch tensor conversion
    "${TORCH_PATH}/lib/libtorch_python.so"
)

# Install the Python module to the appropriate location
install(TARGETS fastloader 
        LIBRARY DESTINATION "${Python3_SITEARCH}"
        ARCHIVE DESTINATION "${Python3_SITEARCH}")