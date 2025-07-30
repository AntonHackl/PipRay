import csv
import shapely.wkt
from shapely.geometry import Point

def load_polygons(wkt_file):
    """Load polygons from WKT file."""
    polygons = []
    with open(wkt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if 'POLYGON' in line or 'MULTIPOLYGON' in line:
                try:
                    geom = shapely.wkt.loads(line)
                    if geom.geom_type == 'Polygon':
                        polygons.append(geom)
                    elif geom.geom_type == 'MultiPolygon':
                        polygons.extend(list(geom.geoms))
                except shapely.errors.WKTReadingError as e:
                    print(f"Error parsing WKT line: {line}. Error: {e}")
                    polygons.append(None)
                    continue
    return polygons

def load_ray_results(ray_csv):
    """Load ray results from CSV file."""
    results = []
    with open(ray_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            point_id = int(row['pointId'])
            polygon_id = int(row['polygonId']) if row['polygonId'] != '' else -1
            results.append({'pointId': point_id, 'polygonId': polygon_id})
    return results

def load_point_coordinates(point_file):
    """Load point coordinates from file."""
    points = []
    with open(point_file, 'r') as f:
        for line in f:
            geom = shapely.wkt.loads(line)
            if geom.geom_type == 'Point':
                points.append(geom)
    return points

def validate_ray_results(polygons, points, ray_results):
    """Validate ray results using point-in-polygon tests."""
    mismatches = 0
    total_points = len(points)
    
    print(f"Validating {total_points} points against {len(polygons)} polygons...")
    
    for i, (point_coords, ray_result) in enumerate(zip(points, ray_results)):
        point = Point(point_coords)
        expected_polygon_id = ray_result['polygonId']
        
        # Find which polygon actually contains this point
        actual_polygon_id = -1
        if expected_polygon_id == -1:
            for j, polygon in enumerate(polygons):
                if polygon.contains(point):
                    actual_polygon_id = j
                    break
            if actual_polygon_id != -1:
                print(f"Point {i} at {point_coords}: Ray missed, but point is inside polygon {actual_polygon_id}")
        else:
            if not polygons[expected_polygon_id].contains(point):
                print(f"Point {i} at {point_coords}: Ray hit polygon {expected_polygon_id}, but point is outside")
            
        
        # Compare expected vs actual
        # if expected_polygon_id != actual_polygon_id:
        #     if expected_polygon_id == -1:
        #         print(f"Point {i} at {point_coords}: Ray missed, but point is inside polygon {actual_polygon_id}")
        #     elif actual_polygon_id == -1:
        #         print(f"Point {i} at {point_coords}: Ray hit polygon {expected_polygon_id}, but point is outside all polygons")
        #     else:
        #         print(f"Point {i} at {point_coords}: Ray hit polygon {expected_polygon_id}, but point is actually in polygon {actual_polygon_id}")
        #     mismatches += 1
    
    return mismatches

def main():
    # Configuration - adjust these file paths
    wkt_file = 'dtl_cnty.wkt'  # Your polygon file
    ray_csv = 'ray_results.csv'  # Your ray results
    point_file = 'dtl_cnty_points.wkt'  # Your point coordinates file
    
    try:
        # Load data
        print("Loading polygons...")
        polygons = load_polygons(wkt_file)
        print(f"Loaded {len(polygons)} polygons")
        
        print("Loading ray results...")
        ray_results = load_ray_results(ray_csv)
        print(f"Loaded {len(ray_results)} ray results")
        
        print("Loading point coordinates...")
        points = load_point_coordinates(point_file)
        print(f"Loaded {len(points)} points")
        
        # Validate
        mismatches = validate_ray_results(polygons, points, ray_results)
        
        # Summary
        print(f"\n=== Validation Summary ===")
        print(f"Total points: {len(points)}")
        print(f"Mismatches: {mismatches}")
        print(f"Accuracy: {((len(points) - mismatches) / len(points) * 100):.2f}%")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please check your file paths and ensure all required files exist.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 