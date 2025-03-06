add_library(prefetcher_lib
    BufferManager.cpp
    PrefetchManager.cpp
)

target_include_directories(prefetcher_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Set C++17 as the standard
set_target_properties(prefetcher_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Install the library
install(TARGETS prefetcher_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib) 