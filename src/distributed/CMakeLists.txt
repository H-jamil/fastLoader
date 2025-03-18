# Create distributed library
add_library(distributed_lib STATIC
    DistributedManager.cpp
)

target_include_directories(distributed_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Link against MPI
target_link_libraries(distributed_lib PUBLIC ${MPI_LIBRARIES})

# Set C++17 as the standard and enable position-independent code
set_target_properties(distributed_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Create test executable
add_executable(test_distributed_manager
    test_distributed_manager.cpp
)

target_link_libraries(test_distributed_manager PRIVATE distributed_lib)

target_include_directories(test_distributed_manager PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install the test binary and library
install(TARGETS test_distributed_manager distributed_lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib) 