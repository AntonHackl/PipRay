# OptiX Raytracer - Single Ray Test

This is a simple OptiX raytracing program that demonstrates casting a single ray from (0,0,0) to (0,0,1) and testing intersection with a triangle defined by vertices (2,2,2), (3,3,3), and (3,2,2).

## Features

- **Single Ray Casting**: Casts one ray from origin (0,0,0) in direction (0,0,1)
- **Triangle Geometry**: Creates a single triangle with specified vertices
- **Bounding Volume Hierarchy (BVH)**: OptiX automatically builds an acceleration structure
- **Intersection Testing**: Reports whether the ray hits the triangle and provides hit information

## Prerequisites

### Required Software
1. **CUDA Toolkit** (version 11.0 or later)
2. **OptiX SDK** (version 7.0 or later)
3. **CMake** (version 3.18 or later)
4. **Visual Studio** (on Windows) or **GCC/Clang** (on Linux)

### Installation

#### CUDA Toolkit
Download and install from: https://developer.nvidia.com/cuda-downloads

#### OptiX SDK
1. Download OptiX SDK from: https://developer.nvidia.com/optix
2. Extract to a directory (e.g., `C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0/`)
3. Set environment variable `OptiX_DIR` to point to the SDK's CMake directory

## Building the Project

### Windows (Visual Studio)

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd OptiXRaytracer
   ```

2. **Set up environment variables**
   ```cmd
   set OptiX_DIR=C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0\SDK\CMake
   ```

3. **Create build directory and configure**
   ```cmd
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019" -A x64
   ```

4. **Build the project**
   ```cmd
   cmake --build . --config Release
   ```

### Linux

1. **Set up environment variables**
   ```bash
   export OptiX_DIR=/path/to/optix/sdk/SDK/CMake
   ```

2. **Create build directory and configure**
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. **Build the project**
   ```bash
   make
   ```

## Running the Program

After successful compilation, run the executable:

```bash
# Windows
./bin/Release/OptiXRaytracer.exe

# Linux
./bin/OptiXRaytracer
```

## Expected Output

The program will output:
```
OptiX Raytracer - Single Ray Test
Casting ray from (0,0,0) to (0,0,1)...
Triangle vertices: (2,2,2), (3,3,3), (3,2,2)
Ray missed - no intersection found
Ray tracing completed!
```

**Note**: The ray from (0,0,0) to (0,0,1) will miss the triangle because:
- The ray travels along the z-axis from z=0 to z=1
- The triangle is positioned at z=2 and z=3
- There is no intersection between the ray path and the triangle's z-coordinates

## Program Versions

### Simple Version (`src/simple_optix.cpp`)
- Uses empty PTX shaders that do nothing
- Demonstrates OptiX setup and pipeline creation
- Good for learning the basic structure
- **Note**: This version doesn't actually perform raytracing

### Working Version (`src/final_optix.cpp`) - **RECOMMENDED**
- Uses actual PTX shaders with raytracing logic
- Performs real ray-triangle intersection testing
- Reports hit/miss results with detailed information
- **Note**: This version actually does the raytracing work

### CUDA Kernels (`src/raytracing.cu`)
- Contains CUDA kernels that could be compiled to PTX
- Currently not used but available for reference
- Shows how to write raytracing kernels in CUDA

## Program Structure

### Key Components

1. **Geometry Setup**: Creates a triangle with vertices (2,2,2), (3,3,3), (3,2,2)
2. **Acceleration Structure**: OptiX builds a BVH for efficient ray-triangle intersection
3. **Ray Generation**: Casts a single ray from (0,0,0) in direction (0,0,1)
4. **Intersection Testing**: Uses OptiX's built-in ray-triangle intersection
5. **Result Reporting**: Prints whether the ray hit or missed the triangle

### Files

- `src/simple_optix.cpp`: Simple version with empty PTX shaders (for demonstration)
- `src/final_optix.cpp`: Working version with actual raytracing PTX shaders (recommended)
- `src/raytracing.cu`: CUDA kernels (not currently used, but available for reference)
- `CMakeLists_simple.txt`: Build configuration for simple version
- `CMakeLists_working.txt`: Build configuration for working version
- `README.md`: This documentation

## Troubleshooting

### Common Issues

1. **OptiX not found**: Ensure `OptiX_DIR` is set correctly
2. **CUDA not found**: Install CUDA Toolkit and ensure it's in PATH
3. **Compilation errors**: Check that you have a compatible GPU and drivers

### GPU Requirements

- NVIDIA GPU with compute capability 6.0 or higher
- Latest NVIDIA drivers
- CUDA-compatible GPU

## Technical Details

### Ray-Triangle Intersection

The program uses OptiX's built-in ray-triangle intersection algorithm, which:
- Automatically handles edge cases
- Provides barycentric coordinates for hit points
- Supports multiple triangle formats

### Acceleration Structure

OptiX automatically builds a BVH (Bounding Volume Hierarchy) that:
- Organizes triangles in a spatial data structure
- Enables efficient ray traversal
- Optimizes intersection testing

### Performance

This simple example demonstrates:
- OptiX initialization and setup
- Geometry upload to GPU
- Ray tracing pipeline creation
- Single ray casting and intersection testing

## Extending the Program

To modify the program for different scenarios:

1. **Change ray origin/direction**: Modify the `raygen_record.data` values
2. **Add more triangles**: Extend the `vertices` and `indices` vectors
3. **Multiple rays**: Change the launch dimensions in `optixLaunch`
4. **Different geometry**: Use other OptiX geometry types (spheres, curves, etc.)

## License

This project is provided as educational material for learning OptiX raytracing. 