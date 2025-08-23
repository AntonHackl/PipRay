#!/usr/bin/env bash
set -e

# Configuration
QUERY_EXEC="/workspace/RayJoin/build/bin/query_exec" # You can modify this as needed

# List of file names to iterate over
# You can modify this list or read from a file
FILE_NAMES=(
  "dtl_cnty"
  "parks_Europe"
  "USACensusBlockGroupBoundaries" 
  "USADetailedWaterBodies"
  # "lakes.bz2"
  # "parks.bz2"
)

# Function to execute query for a single file
execute_query() {
  local filename=$1
  local poly1_path="/root/media/PPoPPAE/datasets/polygons/${filename}.cdb"  # Modify this path as needed
  local poly2_path="/root/media/PPoPPAE/datasets/queries/point-contains_queries_100000/${filename}.wkt" 
  
  echo "Processing file: ${filename}"
  echo "Poly1 path: ${poly1_path}"
  echo "Poly2 path: ${poly2_path}"
  echo "----------------------------------------"
  
  # Execute the query_exec binary
  $QUERY_EXEC \
      -poly1 "${poly1_path}" \
      -poly2 "${poly2_path}" \
      -query "pip" \
      -mode "rt"
  
  echo "Completed processing: ${filename}"
  echo "========================================"
  echo ""
}

# Main execution loop
echo "Starting query execution for ${#FILE_NAMES[@]} files..."
echo "Query executable: ${QUERY_EXEC}"
echo ""

# Check if query_exec exists
if [[ ! -f "${QUERY_EXEC}" ]]; then
  echo "Error: query_exec binary not found at ${QUERY_EXEC}"
  echo "Please ensure the RayJoin project is built and the binary exists."
  exit 1
fi

# Iterate over each file name
for filename in "${FILE_NAMES[@]}"; do
  execute_query "$filename"
done

echo "All queries completed successfully!" 