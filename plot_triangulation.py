#!/usr/bin/env python3
"""
Script to visualize the fifth polygon from dtl_cnty dataset and its Constrained Delaunay Triangulation.

Constrained Delaunay Triangulation (CDT) is a triangulation that:
1. Preserves all input boundary segments as edges in the triangulation
2. Maximizes the minimum angle of triangles (Delaunay property) where possible
3. Respects holes and complex polygon boundaries

This script uses the 'triangle' library which implements Shewchuk's Triangle algorithm,
a robust implementation of CDT.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np
from shapely.wkt import loads
from shapely.geometry import Polygon, MultiPolygon
import triangle


def read_wkt_file(filename):
    """Read WKT geometries from file."""
    geometries = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        geom = loads(line)
                        geometries.append(geom)
                    except Exception as e:
                        print(f"Error parsing line: {line[:100]}... Error: {e}")
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return []
    return geometries


def polygon_to_triangle_input(polygon):
    """Convert a Shapely polygon to triangle library input format."""
    # Get exterior coordinates
    exterior_coords = list(polygon.exterior.coords[:-1])  # Remove duplicate last point
    vertices = np.array(exterior_coords)
    
    # Create segments for exterior boundary
    n_exterior = len(exterior_coords)
    exterior_segments = [(i, (i + 1) % n_exterior) for i in range(n_exterior)]
    
    segments = exterior_segments
    holes = []
    
    # Add interior holes if any
    vertex_offset = n_exterior
    for interior in polygon.interiors:
        interior_coords = list(interior.coords[:-1])  # Remove duplicate last point
        hole_vertices = np.array(interior_coords)
        vertices = np.vstack([vertices, hole_vertices])
        
        # Add hole segments
        n_hole = len(interior_coords)
        hole_segments = [(vertex_offset + i, vertex_offset + (i + 1) % n_hole) 
                        for i in range(n_hole)]
        segments.extend(hole_segments)
        
        # Add hole point (centroid of hole)
        hole_centroid = np.mean(hole_vertices, axis=0)
        holes.append(hole_centroid.tolist())
        
        vertex_offset += n_hole
    
    return {
        'vertices': vertices,
        'segments': segments,
        'holes': holes
    }


def triangulate_polygon(polygon):
    """Triangulate a polygon using Constrained Delaunay Triangulation.
    
    The triangle library performs CDT when given segments (constraints).
    The 'p' flag tells it to triangulate a Planar Straight Line Graph (PSLG)
    which respects the boundary constraints we provide.
    """
    triangle_input = polygon_to_triangle_input(polygon)
    
    # Prepare input for triangle library
    tri_input = {
        'vertices': triangle_input['vertices'],
        'segments': triangle_input['segments']  # These are the constraints for CDT
    }
    
    if triangle_input['holes']:
        tri_input['holes'] = triangle_input['holes']
    
    # Perform Constrained Delaunay Triangulation
    # 'p' = Planar Straight Line Graph (respects segment constraints - this makes it CDT)
    # 'z' = zero-based indexing
    # 'q' = quiet (no output)
    # 'D' = Delaunay (ensures Delaunay property where possible)
    try:
        result = triangle.triangulate(tri_input, 'pzqD')
        return result
    except Exception as e:
        print(f"CDT triangulation failed: {e}")
        return None


def plot_polygon_and_triangulation(polygon, triangulation, polygon_index):
    """Plot the original polygon and its triangulation side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original polygon
    ax1.set_title(f'Original Polygon #{polygon_index}')
    ax1.set_aspect('equal')
    
    # Plot exterior
    x, y = polygon.exterior.xy
    ax1.plot(x, y, 'b-', linewidth=2, label='Exterior')
    ax1.fill(x, y, alpha=0.3, color='lightblue')
    
    # Plot holes
    for i, interior in enumerate(polygon.interiors):
        x, y = interior.xy
        ax1.plot(x, y, 'r-', linewidth=2, label='Hole' if i == 0 else "")
        ax1.fill(x, y, alpha=1.0, color='white')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot triangulation
    if triangulation is not None:
        ax2.set_title(f'CDT Triangulated Polygon #{polygon_index}')
        ax2.set_aspect('equal')
        
        vertices = triangulation['vertices']
        triangles = triangulation['triangles']
        
        # Plot triangles
        for tri in triangles:
            triangle_coords = vertices[tri]
            triangle_patch = patches.Polygon(triangle_coords, 
                                           fill=False, 
                                           edgecolor='red', 
                                           linewidth=1)
            ax2.add_patch(triangle_patch)
        
        # Plot vertices
        ax2.scatter(vertices[:, 0], vertices[:, 1], 
                   c='blue', s=20, zorder=5, label='Vertices')
        
        # Plot boundary edges in bold
        segments = triangulation.get('segments', [])
        if len(segments) > 0:
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
        print(f"  Vertices: {len(vertices)}")
        print(f"  Triangles: {len(triangles)}")
        print(f"  Boundary segments (constraints): {len(segments) if segments is not None else 0}")
        
        # Verify CDT properties
        boundary_edges = set()
        if segments is not None:
            for seg in segments:
                boundary_edges.add(tuple(sorted(seg)))
        
        triangle_edges = set()
        for tri in triangles:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                triangle_edges.add(edge)
        
        constraints_preserved = boundary_edges.issubset(triangle_edges)
        print(f"  All boundary constraints preserved: {constraints_preserved}")
        if not constraints_preserved:
            missing = boundary_edges - triangle_edges
            print(f"  Missing constraint edges: {len(missing)}")
    else:
        ax2.text(0.5, 0.5, 'Triangulation Failed', 
                transform=ax2.transAxes, 
                ha='center', va='center',
                fontsize=16, color='red')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    dataset_file = "dtl_cnty.wkt"
    target_polygon_index = 5  # Fifth polygon (0-indexed)
    
    print(f"Loading geometries from {dataset_file}...")
    geometries = read_wkt_file(dataset_file)
    
    if not geometries:
        print("No geometries found!")
        return
    
    print(f"Found {len(geometries)} geometries")
    
    if len(geometries) <= target_polygon_index:
        print(f"Dataset only contains {len(geometries)} geometries, "
              f"but requested polygon #{target_polygon_index + 1}")
        return
    
    # Get the target geometry
    target_geom = geometries[target_polygon_index]
    
    # Handle MultiPolygon by taking the first polygon
    if isinstance(target_geom, MultiPolygon):
        if len(target_geom.geoms) > 0:
            polygon = target_geom.geoms[0]
            print(f"MultiPolygon found, using first polygon out of {len(target_geom.geoms)}")
        else:
            print("Empty MultiPolygon!")
            return
    elif isinstance(target_geom, Polygon):
        polygon = target_geom
    else:
        print(f"Geometry #{target_polygon_index + 1} is not a polygon: {type(target_geom)}")
        return
    
    print(f"Processing polygon #{target_polygon_index + 1}:")
    print(f"  Exterior vertices: {len(polygon.exterior.coords)}")
    print(f"  Interior holes: {len(polygon.interiors)}")
    print(f"  Area: {polygon.area:.2f}")
    print(f"  Bounds: {polygon.bounds}")
    
    # Triangulate the polygon
    print("Performing Constrained Delaunay Triangulation...")
    triangulation = triangulate_polygon(polygon)
    
    # Plot results
    plot_polygon_and_triangulation(polygon, triangulation, target_polygon_index + 1)


if __name__ == "__main__":
    main()
