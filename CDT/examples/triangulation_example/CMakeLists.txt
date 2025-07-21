cmake_minimum_required(VERSION 3.10)
project(triangulation_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Program Files (x86) to the search path
list(APPEND CMAKE_PREFIX_PATH "C:/Program Files (x86)/CDT")

# Find the CDT package
find_package(CDT REQUIRED)

# Add the executable
add_executable(triangulation_example main.cpp)

# Link against CDT
target_link_libraries(triangulation_example PRIVATE CDT::CDT) 