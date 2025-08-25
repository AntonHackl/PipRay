#!/usr/bin/env python3
"""
Script to visualize a single polygon and its Constrained Delaunay Triangulation.

This script reads the output from the C++ triangulation program (single_polygon_triangulation.txt)
and displays both the original polygon and its triangulation side by side.

The C++ program uses the CDT (Constrained Delaunay Triangulation) library to perform
the triangulation, which ensures that:
1. All boundary segments are preserved as edges in the triangulation
2. The Delaunay property is maximized where possible
3. Holes and complex polygon boundaries are respected
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import numpy as np
import random
import sys


def load_triangulation_data(filename):
    """Load triangulation data from text file."""
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Please run the C++ triangulation program first:")
        print("  ./triangulate_single_polygon dtl_cnty.wkt 5")
        return None
    
    data = {}
    line_index = 0
    
    # Skip comments and read polygon info
    while line_index < len(lines) and lines[line_index].startswith('#'):
        line_index += 1
    if line_index >= len(lines):
        return None
    
    # Read polygon index and number
    parts = lines[line_index].strip().split()
    data['polygon_index'] = int(parts[0])
    data['polygon_number'] = int(parts[1])
    line_index += 1
    
    # Read original polygon exterior
    while line_index < len(lines) and lines[line_index].startswith('#'):
        line_index += 1
    if line_index >= len(lines):
        return None
    
    num_exterior_vertices = int(lines[line_index].strip())
    line_index += 1
    
    exterior_vertices = []
    for i in range(num_exterior_vertices):
        while line_index < len(lines) and lines[line_index].startswith('#'):
            line_index += 1
        if line_index >= len(lines):
            return None
        
        parts = lines[line_index].strip().split()
        exterior_vertices.append({'x': float(parts[0]), 'y': float(parts[1])})
        line_index += 1
    
    data['original_polygon'] = {'exterior': exterior_vertices, 'holes': []}
    
    # Read holes
    while line_index < len(lines) and lines[line_index].startswith('#'):
        line_index += 1
    if line_index >= len(lines):
        return None
    
    num_holes = int(lines[line_index].strip())
    line_index += 1
    
    for hole_idx in range(num_holes):
        while line_index < len(lines) and lines[line_index].startswith('#'):
            line_index += 1
        if line_index >= len(lines):
            return None
        
        num_hole_vertices = int(lines[line_index].strip())
        line_index += 1
        
        hole_vertices = []
        for i in range(num_hole_vertices):
            while line_index < len(lines) and lines[line_index].startswith('#'):
                line_index += 1
            if line_index >= len(lines):
                return None
            
            parts = lines[line_index].strip().split()
            hole_vertices.append({'x': float(parts[0]), 'y': float(parts[1])})
            line_index += 1
        
        data['original_polygon']['holes'].append(hole_vertices)
    
    # Read triangulation vertices
    while line_index < len(lines) and lines[line_index].startswith('#'):
        line_index += 1
    if line_index >= len(lines):
        return None
    
    num_triangulation_vertices = int(lines[line_index].strip())
    line_index += 1
    
    triangulation_vertices = []
    for i in range(num_triangulation_vertices):
        while line_index < len(lines) and lines[line_index].startswith('#'):
            line_index += 1
        if line_index >= len(lines):
            return None
        
        parts = lines[line_index].strip().split()
        triangulation_vertices.append({'x': float(parts[0]), 'y': float(parts[1])})
        line_index += 1
    
    # Read triangles
    while line_index < len(lines) and lines[line_index].startswith('#'):
        line_index += 1
    if line_index >= len(lines):
        return None
    
    num_triangles = int(lines[line_index].strip())
    line_index += 1
    
    triangles = []
    for i in range(num_triangles):
        while line_index < len(lines) and lines[line_index].startswith('#'):
            line_index += 1
        if line_index >= len(lines):
            return None
        
        parts = lines[line_index].strip().split()
        # Parse coordinates: v1_x v1_y v2_x v2_y v3_x v3_y
        if len(parts) >= 6:
            triangle = [
                [float(parts[0]), float(parts[1])],  # vertex 1
                [float(parts[2]), float(parts[3])],  # vertex 2
                [float(parts[4]), float(parts[5])]   # vertex 3
            ]
            triangles.append(triangle)
        line_index += 1
    
    # Read segments
    while line_index < len(lines) and lines[line_index].startswith('#'):
        line_index += 1
    if line_index >= len(lines):
        return None
    
    num_segments = int(lines[line_index].strip())
    line_index += 1
    
    segments = []
    for i in range(num_segments):
        while line_index < len(lines) and lines[line_index].startswith('#'):
            line_index += 1
        if line_index >= len(lines):
            return None
        
        parts = lines[line_index].strip().split()
        segments.append([int(parts[0]), int(parts[1])])
        line_index += 1
    
    data['triangulation'] = {
        'vertices': triangulation_vertices,
        'triangles': triangles,
        'segments': segments
    }
    
    return data


def create_polygon_from_data(polygon_data):
    """Create a matplotlib polygon patch from the polygon data."""
    # Extract exterior coordinates
    exterior_coords = []
    for point in polygon_data['exterior']:
        exterior_coords.append([point['x'], point['y']])
    
    # Create the polygon patch
    polygon_patch = patches.Polygon(exterior_coords, 
                                   fill=False, 
                                   edgecolor='blue', 
                                   linewidth=2,
                                   label='Exterior')
    
    # Create hole patches
    hole_patches = []
    for i, hole in enumerate(polygon_data['holes']):
        hole_coords = []
        for point in hole:
            hole_coords.append([point['x'], point['y']])
        
        hole_patch = patches.Polygon(hole_coords, 
                                   fill=True, 
                                   facecolor='white',
                                   edgecolor='red', 
                                   linewidth=2,
                                   label='Hole' if i == 0 else "")
        hole_patches.append(hole_patch)
    
    return polygon_patch, hole_patches


def plot_polygon_and_triangulation(data):
    """Plot the original polygon and its triangulation side by side."""
    if data is None:
        return
    
    polygon_number = data.get('polygon_number', 'Unknown')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original polygon
    ax1.set_title(f'Original Polygon #{polygon_number}')
    ax1.set_aspect('equal')
    
    # Create and add the main polygon
    polygon_patch, hole_patches = create_polygon_from_data(data['original_polygon'])
    ax1.add_patch(polygon_patch)
    
    # Add holes
    for hole_patch in hole_patches:
        ax1.add_patch(hole_patch)
    
    # Fill the polygon with light blue
    exterior_coords = np.array([[point['x'], point['y']] for point in data['original_polygon']['exterior']])
    ax1.fill(exterior_coords[:, 0], exterior_coords[:, 1], alpha=0.3, color='lightblue')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot triangulation
    triangulation = data['triangulation']
    ax2.set_title(f'CDT Triangulated Polygon #{polygon_number}')
    ax2.set_aspect('equal')
    
    # Extract vertices and triangles
    vertices = np.array([[point['x'], point['y']] for point in triangulation['vertices']])
    triangles = triangulation['triangles']
    
    # Plot triangles (triangles now contain actual coordinates, not indices)
    # Generate random colors for each triangle
    random.seed(42)  # For reproducible colors
    triangle_colors = []
    for i in range(len(triangles)):
        # Generate random RGB values
        color = (random.random(), random.random(), random.random())
        triangle_colors.append(color)
    
    for i, triangle in enumerate(triangles):
        triangle_coords = np.array(triangle)  # triangle is already coordinates
        triangle_patch = patches.Polygon(triangle_coords, 
                                       fill=True,
                                       facecolor=triangle_colors[i],
                                       edgecolor='black', 
                                       linewidth=1.5,
                                       alpha=0.6,
                                       label=f'Triangle {i+1}' if i < 5 else "")  # Label first 5 triangles only
        ax2.add_patch(triangle_patch)
    
    # Plot vertices
    ax2.scatter(vertices[:, 0], vertices[:, 1], 
               c='white', s=30, edgecolors='black', linewidth=1.5, zorder=10, label='Vertices')
    
    # Plot boundary edges (constraints) in bold
    segments = triangulation.get('segments', [])
    if segments:
        boundary_lines = [vertices[seg] for seg in segments]
        boundary_collection = LineCollection(boundary_lines, 
                                            colors='black', 
                                            linewidths=3, 
                                            label='Boundary')
        ax2.add_collection(boundary_collection)
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Print statistics
    print(f"Constrained Delaunay Triangulation statistics:")
    print(f"  Polygon number: {polygon_number}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(triangles)}")
    print(f"  Boundary segments (constraints): {len(segments)}")
    
    # Verify CDT properties
    boundary_edges = set()
    for seg in segments:
        boundary_edges.add(tuple(sorted(seg)))
    
    # For triangle edges, we need to convert coordinates back to indices
    triangle_edges = set()
    for triangle in triangles:
        # Find indices of triangle vertices in the vertices array
        triangle_indices = []
        for vertex_coord in triangle:
            # Find matching vertex index
            for i, vertex in enumerate(vertices):
                if abs(vertex[0] - vertex_coord[0]) < 1e-6 and abs(vertex[1] - vertex_coord[1]) < 1e-6:
                    triangle_indices.append(i)
                    break
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    txt_file = "single_polygon_triangulation.txt"
    
    if len(sys.argv) > 1:
        txt_file = sys.argv[1]
    
    print(f"Loading triangulation data from: {txt_file}")
    data = load_triangulation_data(txt_file)
    
    if data is None:
        return
    
    print("Data loaded successfully!")
    plot_polygon_and_triangulation(data)


if __name__ == "__main__":
    main()
