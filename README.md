# OptiX Raytracer - Single Ray Test

This is a simple OptiX raytracing program.

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
## Extending the Program

To modify the program for different scenarios:

1. **Change ray origin/direction**: Modify the `raygen_record.data` values
2. **Add more triangles**: Extend the `vertices` and `indices` vectors
3. **Multiple rays**: Change the launch dimensions in `optixLaunch`
4. **Different geometry**: Use other OptiX geometry types (spheres, curves, etc.)