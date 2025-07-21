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

// Function to read polygon vertices from a WKT file
std::vector<std::vector<CDT::V2d<float>>> readPolygonVerticesFromFile(const std::string& filepath);

// Function to triangulate a single polygon
CDT::TriangleVec triangulatePolygon(const std::vector<CDT::V2d<float>>& polygon);

// Function to count total triangles from triangulated polygons
size_t countTriangles(const std::vector<CDT::TriangleVec>& triangulated_polygons);
