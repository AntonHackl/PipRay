#!/usr/bin/env python3
"""
Script to generate uniformly random points within the bounding box of polygons in a WKT file.
Usage: python generate_random_points.py <wkt_file> <num_points>
"""

import sys
import re
import random
from shapely.geometry import Polygon
from shapely.wkt import loads
import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def parse_wkt_file(filename):
    """
    Parse a WKT file and extract all POLYGON geometries.
    Returns a list of Shapely Polygon objects.
    """
    polygons = []
    
    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
                
            if 'POLYGON' in line and 'MULTI' not in line:
                try:
                    # Parse the WKT string
                    polygon = loads(line)
                    if polygon.is_valid:
                        polygons.append(polygon)
                    else:
                        print(f"Warning: Invalid polygon on line {line_num}, skipping")
                except Exception as e:
                    print(f"Warning: Could not parse polygon on line {line_num}: {e}")
                    continue
    
    return polygons

def calculate_bounding_box(polygons):
    """
    Calculate the bounding box that encompasses all polygons.
    Returns (min_x, min_y, max_x, max_y)
    """
    if not polygons:
        raise ValueError("No valid polygons found")
    
    # Initialize with the first polygon's bounds
    min_x, min_y, max_x, max_y = polygons[0].bounds
    
    # Expand bounds to include all polygons
    for polygon in polygons[1:]:
        bounds = polygon.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])
    
    return min_x, min_y, max_x, max_y

def generate_random_points(min_x, min_y, max_x, max_y, num_points):
    """
    Generate uniformly random points within the bounding box.
    Returns a list of (x, y) tuples.
    """
    points = []
    for _ in range(num_points):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        points.append((x, y))
    
    return points

def main():
    parser = argparse.ArgumentParser(
        description='Generate uniformly random points within the bounding box of polygons in a WKT file'
    )
    parser.add_argument('wkt_file', help='Path to the WKT file')
    parser.add_argument('num_points', type=int, help='Number of random points to generate')
    parser.add_argument('--output', '-o', help='Output file for points (default: stdout)')
    parser.add_argument('--format', choices=['wkt', 'csv', 'xy'], default='wkt',
                       help='Output format: wkt (POINT), csv (x,y), or xy (x y)')
    parser.add_argument('--show', action='store_true', help='Show a plot of the polygons and generated points')
    
    args = parser.parse_args()
    

    try:
        print(f"Reading polygons from {args.wkt_file}...")
        polygons = parse_wkt_file(args.wkt_file)
        
        if not polygons:
            print("Error: No valid polygons found in the WKT file")
            sys.exit(1)
        
        print(f"Found {len(polygons)} valid polygons")
        
        # min_x, min_y, max_x, max_y = calculate_bounding_box(polygons)
        min_x, min_y, max_x, max_y = (-120, 30, -80, 50)
        print(f"Bounding box: ({min_x:.6f}, {min_y:.6f}) to ({max_x:.6f}, {max_y:.6f})")
        print(f"Bounding box dimensions: {max_x - min_x:.6f} x {max_y - min_y:.6f}")
        
        print(f"Generating {args.num_points} random points...")
        points = generate_random_points(min_x, min_y, max_x, max_y, args.num_points)

        if args.output:
            with open(args.output, 'w') as f:
                for point in points:
                    if args.format == 'wkt':
                        f.write(f"POINT({point[0]:.6f} {point[1]:.6f})\n")
                    elif args.format == 'csv':
                        f.write(f"{point[0]:.6f},{point[1]:.6f}\n")
                    elif args.format == 'xy':
                        f.write(f"{point[0]:.6f} {point[1]:.6f}\n")
            print(f"Points written to {args.output}")
        else:
            for point in points:
                if args.format == 'wkt':
                    print(f"POINT({point[0]:.6f} {point[1]:.6f})")
                elif args.format == 'csv':
                    print(f"{point[0]:.6f},{point[1]:.6f}")
                elif args.format == 'xy':
                    print(f"{point[0]:.6f} {point[1]:.6f}")

        if args.show:
            _, ax = plt.subplots(figsize=(8, 8))
            for poly in polygons:
                if poly.is_empty:
                    continue
                if poly.geom_type == 'Polygon':
                    mpl_poly = MplPolygon(list(poly.exterior.coords), closed=True, edgecolor='blue', facecolor='none', lw=1)
                    ax.add_patch(mpl_poly)
                    for interior in poly.interiors:
                        hole = MplPolygon(list(interior.coords), closed=True, edgecolor='red', facecolor='none', lw=1, ls='--')
                        ax.add_patch(hole)
            # xs, ys = zip(*points)
            # ax.scatter(xs, ys, color='orange', s=10, label='Random Points')
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal', 'box')
            ax.set_title('Polygons and Random Points')
            ax.legend()
            plt.show()

        print("Done!")

    except FileNotFoundError:
        print(f"Error: File '{args.wkt_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 