#!/bin/bash

# Batch processing script for PipRay raytracing performance tests
# This script runs the raytracer with different polygon and point datasets

# Configuration
INSTALL_DIR="/root/media/Spatial_Data_Management/PipRay/build"
RAYTRACER_EXE="${INSTALL_DIR}/bin/raytracer"
RESULTS_DIR="/root/media/Spatial_Data_Management/pipray_results"
NUM_RUNS=10

# Base paths for datasets
POLYGON_BASE_PATH="/root/media/PPoPPAE/datasets/polygons"
POINT_BASE_PATH="/root/media/PPoPPAE/datasets/queries/point-contains_queries_100000"

FILE_NAMES=(
  "dtl_cnty"
#   "parks_Europe"
#   "USACensusBlockGroupBoundaries" 
#   "USADetailedWaterBodies"
  "lakes.bz2"
  "parks.bz2"
)

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# Function to run a single test
run_test() {
    local filename="$1"
    local polygon_dataset="${POLYGON_BASE_PATH}/${filename}.wkt"
    local point_dataset="${POINT_BASE_PATH}/${filename}.wkt"
    local test_name="${filename}_pip_test"
    local output_json="${RESULTS_DIR}/${test_name}.json"
    
    echo "Running test: ${test_name}"
    echo "  Filename: ${filename}"
    echo "  Polygon dataset: ${polygon_dataset}"
    echo "  Point dataset: ${point_dataset}"
    echo "  Output: ${output_json}"
    
    # Check if input files exist
    if [ ! -f "${polygon_dataset}" ]; then
        echo "  ⚠ Warning: Polygon dataset not found: ${polygon_dataset}"
        return 1
    fi
    if [ ! -f "${point_dataset}" ]; then
        echo "  ⚠ Warning: Point dataset not found: ${point_dataset}"
        return 1
    fi
    
    # Run the raytracer with all parameters
    "${RAYTRACER_EXE}" \
        --dataset "${polygon_dataset}" \
        --points "${point_dataset}" \
        --output "${output_json}" \
        --runs "${NUM_RUNS}"
    
    local exit_code=$?
    if [ ${exit_code} -eq 0 ]; then
        echo "  ✓ Test completed successfully"
        return 0
    else
        echo "  ✗ Test failed with exit code ${exit_code}"
        return 1
    fi
}

# Function to generate test name
generate_test_name() {
    local filename="$1"
    echo "${filename}_pip_test"
}

# Main execution
echo "=== PipRay Batch Processing Script ==="
echo "Installation directory: ${INSTALL_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Polygon base path: ${POLYGON_BASE_PATH}"
echo "Point base path: ${POINT_BASE_PATH}"
echo "Number of runs per test: ${NUM_RUNS}"
echo "Number of files to process: ${#FILE_NAMES[@]}"
echo ""

# Check if raytracer executable exists
if [ ! -f "${RAYTRACER_EXE}" ]; then
    echo "Error: Raytracer executable not found at: ${RAYTRACER_EXE}"
    echo "Please build the project first or update the INSTALL_DIR path."
    exit 1
fi

# Process each file
test_count=0
successful_tests=0
failed_tests=0

for filename in "${FILE_NAMES[@]}"; do
    test_count=$((test_count + 1))
    
    echo "Progress: ${test_count}/${#FILE_NAMES[@]}"
    echo "Processing: ${filename}"
    
    if run_test "${filename}"; then
        successful_tests=$((successful_tests + 1))
    else
        failed_tests=$((failed_tests + 1))
    fi
    echo ""
done

echo "=== Batch processing completed ==="
echo "Total tests attempted: ${test_count}"
echo "Successful tests: ${successful_tests}"
echo "Failed tests: ${failed_tests}"
echo "Results saved to: ${RESULTS_DIR}"
echo ""
echo "You can analyze the results using:"
echo "  python visualize_performance_results.py --directory ${RESULTS_DIR}"
