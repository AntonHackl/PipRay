#!/usr/bin/env python3
"""
Script to extract the first 100,000 polygons from parks_Europe.wkt and save them to parks_europe_small.wkt.

This script reads the parks_Europe.wkt file line by line, parses both POLYGON and MULTIPOLYGON geometries,
and writes the first 100,000 valid polygons to a new WKT file.
"""

import sys
import os
from shapely.wkt import loads


def extract_polygons(input_file, max_polygons=100000):
    """Extract polygons from WKT file, one per line, including from multipolygons."""
    polygons = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse the WKT string using Shapely
                    geometry = loads(line)
                    
                    if geometry.is_valid:
                        if geometry.geom_type == 'Polygon':
                            # Single polygon
                            polygons.append(line)
                        elif geometry.geom_type == 'MultiPolygon':
                            # Extract individual polygons from multipolygon
                            for polygon in geometry.geoms:
                                if polygon.is_valid:
                                    # Convert back to WKT string
                                    polygon_wkt = polygon.wkt
                                    polygons.append(polygon_wkt)
                                    
                                    if len(polygons) >= max_polygons:
                                        break
                        else:
                            print(f"Warning: Unsupported geometry type '{geometry.geom_type}' on line {line_num}, skipping")
                            continue
                            
                        if len(polygons) >= max_polygons:
                            break
                    else:
                        print(f"Warning: Invalid geometry on line {line_num}, skipping")
                        
                except Exception as e:
                    print(f"Warning: Could not parse geometry on line {line_num}: {e}")
                    continue
                        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found!")
        return None
    except Exception as e:
        print(f"Error reading file {input_file}: {e}")
        return None
    
    return polygons


def write_polygons(output_file, polygons):
    """Write polygons to output file, one per line."""
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for polygon in polygons:
                file.write(polygon + '\n')
        
        print(f"Successfully wrote {len(polygons)} polygons to {output_file}")
        return True
    except Exception as e:
        print(f"Error writing file {output_file}: {e}")
        return False


def main():
    """Main function."""
    input_file = "parks_Europe.wkt"
    output_file = "parks_europe_small.wkt"
    max_polygons = 500000
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure the file exists in the current directory.")
        return 1
    
    print(f"Reading polygons from {input_file}...")
    
    # Extract polygons
    polygons = extract_polygons(input_file, max_polygons)
    
    if polygons is None:
        return 1
    
    if not polygons:
        print("Error: No valid polygons found in the WKT file!")
        return 1
    
    print(f"Found {len(polygons)} valid polygons")
    
    # Write to output file
    if write_polygons(output_file, polygons):
        print(f"Successfully created {output_file}")
        print(f"File contains {len(polygons)} polygons")
        
        # Calculate file size
        file_size = os.path.getsize(output_file)
        print(f"Output file size: {file_size / (1024*1024):.2f} MB")
        
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 