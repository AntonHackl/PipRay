#!/bin/bash

echo "Building OptiX Raytracer..."

# Check if OptiX_DIR is set
if [ -z "$OptiX_DIR" ]; then
    echo "Warning: OptiX_DIR environment variable is not set."
    echo "Please set it to your OptiX SDK CMake directory."
    echo "Example: export OptiX_DIR=/path/to/optix/sdk/SDK/CMake"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build the project
echo "Building project..."
make
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build completed successfully!"
echo "Executable location: build/OptiXRaytracer"
cd .. 