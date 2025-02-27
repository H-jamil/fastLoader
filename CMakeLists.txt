   cmake_minimum_required(VERSION 3.10)
   project(DistDataLoader)

   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)
   # Include directories
   include_directories(include)

   # Source files
   set(SOURCES
       src/storage/FileSystemBacked.cpp
       src/storage/test_filesystem_backend.cpp
   )

   # Create the executable
   add_executable(test_filesystem_backend ${SOURCES})