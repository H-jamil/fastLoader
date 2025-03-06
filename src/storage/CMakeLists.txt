# Create storage library
add_library(storage_lib STATIC
    FileSystemBacked.cpp
)

target_include_directories(storage_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Link against filesystem library
if(CMAKE_CXX_STANDARD_LIBRARIES MATCHES "stdc\\+\\+fs")
    target_link_libraries(storage_lib PUBLIC stdc++fs)
endif()

# Set C++17 as the standard
set_target_properties(storage_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Create test executable
add_executable(test_filesystem_backend
    test_filesystem_backend.cpp
)

target_link_libraries(test_filesystem_backend PRIVATE storage_lib)

target_include_directories(test_filesystem_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Install the test binary
install(TARGETS test_filesystem_backend storage_lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib) 