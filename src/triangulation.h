#pragma once

// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <CDT.h>
#include <vector>
#include <string>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/wkt/wkt.hpp>

struct PolygonWithHoles {
    std::vector<CDT::V2d<float>> outer;
    std::vector<std::vector<CDT::V2d<float>>> holes;
};

// Triangulation method statistics
struct TriangulationStats {
    int default_method = 0;      // Method 0: Default CDT
    int try_resolve_method = 0;  // Method 1: TryResolve
    int dont_check_method = 0;   // Method 2: DontCheck
    int failed_method = 0;       // Method 3: All methods failed
    
    void print() const {
        std::cout << "Triangulation methods used: Default=" << default_method 
                  << ", TryResolve=" << try_resolve_method 
                  << ", DontCheck=" << dont_check_method 
                  << ", Failed=" << failed_method << std::endl;
    }
};

// Function to read polygon vertices from a WKT file
std::vector<PolygonWithHoles> readPolygonVerticesFromFile(const std::string& filepath);

// Function to triangulate a single polygon, returns triangles and method used (0=default, 1=tryresolve, 2=dontcheck, 3=failed)
std::pair<CDT::TriangleVec, int> triangulatePolygon(const PolygonWithHoles& polygon);

// Function to count total triangles from triangulated polygons
size_t countTriangles(const std::vector<CDT::TriangleVec>& triangulated_polygons);
