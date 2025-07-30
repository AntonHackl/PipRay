cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "C:/Users/anton/Documents/Uni/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

project(OptiXSingleRay LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add preprocessor definitions to prevent Windows min/max macro conflicts
add_definitions(-DNOMINMAX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(CUDA REQUIRED)
include(cmake/FindOptiX.cmake)

include(cmake/nvcuda_compile_module.cmake)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CUDA_SOURCES ${SRC_DIR}/raytracing.cu)
set(CPP_SOURCE   ${SRC_DIR}/main.cpp)
set(DATASET_LOADER_SOURCE ${SRC_DIR}/dataset_loader.cpp)
set(TRIANGULATE_SOURCE ${SRC_DIR}/triangulate_dataset.cpp)
set(TRIANGULATION_LIB_SOURCE ${SRC_DIR}/triangulation.cpp)

set(PTX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")

NVCUDA_COMPILE_MODULE(
    SOURCES       ${CUDA_SOURCES}
    TARGET_PATH   ${PTX_OUTPUT_DIR}
    EXTENSION     "${OPTIX_MODULE_EXTENSION}"
    GENERATED_FILES PTX_FILES
    NVCC_OPTIONS  "${OPTIX_PROGRAM_TARGET}" "--gpu-architecture=sm_75" "--relocatable-device-code=true" "--expt-relaxed-constexpr" "-I${OptiX_INCLUDE}"
)

add_executable(raytracer ${CPP_SOURCE} ${DATASET_LOADER_SOURCE} ${TRIANGULATION_LIB_SOURCE})

target_include_directories(raytracer PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    ${OptiX_INCLUDE} 
    ${SRC_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/CDT/CDT/include
    C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)

set(PTX_FILE "${PTX_OUTPUT_DIR}/raytracing${OPTIX_MODULE_EXTENSION}")
add_custom_target(copy_ptx DEPENDS ${PTX_FILES})
add_dependencies(raytracer copy_ptx)

# OptiX host functions are loaded through the stub headers at runtime, so linking the explicit
# OptiX libraries is unnecessary and breaks the build when those import libraries are not found.
# RayJoin works without that explicit linkage, so mirror the same approach here.
target_link_libraries(raytracer PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

set_target_properties(raytracer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

# add_executable(triangulate_dataset ${TRIANGULATE_SOURCE} ${TRIANGULATION_LIB_SOURCE})

# target_include_directories(triangulate_dataset PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/CDT/CDT/include
#     C:/Users/anton/Documents/Uni/vcpkg/installed/x64-windows/include)

# set_target_properties(triangulate_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 