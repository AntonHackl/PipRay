// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "triangulation.h"
#include <fstream>
#include <iostream>

using namespace std;

std::vector<std::vector<CDT::V2d<float>>> readPolygonVerticesFromFile(const std::string& filepath)
{
    using BoostPolygon = boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<float>>;
    ifstream file(filepath);
    vector<string> lines;
    string line;
    while(getline(file, line))
    {
        // Find the first line containing "POLYGON"
        if(line.find("POLYGON") != string::npos &&
           line.find("MULTI") == string::npos)
        {
            lines.push_back(line);
        }
    }
    if(lines.empty())
    {
        return {};
    }
    vector<BoostPolygon> polygons =
        vector<BoostPolygon>(lines.size(), BoostPolygon());
    try
    {
        for(size_t i = 0; i < lines.size(); ++i)
        {
            boost::geometry::read_wkt(lines[i], polygons[i]);
        }
    }
    catch(const exception& e)
    {
        cerr << "Error reading WKT: " << e.what() << endl;
        return {};
    }
    vector<vector<CDT::V2d<float>>> polygon_vertices;
    // Only use the exterior ring
    for(const auto& poly : polygons)
    {
        vector<CDT::V2d<float>> vertices;
        
        // Get the outer ring without validation for now to debug
        auto outer = boost::geometry::exterior_ring(poly);
        for(const auto& pt : outer)
        {
            vertices.push_back({pt.x(), pt.y()});
        }
        if(!vertices.empty() && vertices.front().x == vertices.back().x &&
           vertices.front().y == vertices.back().y)
        {
            vertices.pop_back();
        }
        
        if (vertices.size() >= 3) {
            polygon_vertices.push_back(vertices);
        } else {
            cerr << "Polygon has less than 3 vertices after processing." << endl;
        }
    }
    return polygon_vertices;
}

CDT::TriangleVec triangulatePolygon(const std::vector<CDT::V2d<float>>& polygon)
{
    CDT::Triangulation<float> cdt;

    vector<CDT::V2d<float>> vertices = polygon;
    auto duplicates_info = CDT::RemoveDuplicates(vertices);

    vector<CDT::Edge> edges;
    edges.reserve(vertices.size());
    for(size_t i = 0; i < vertices.size(); ++i)
    {
        edges.push_back(CDT::Edge(CDT::VertInd(i), CDT::VertInd((i + 1) % vertices.size())));
    }

    cdt.insertVertices(vertices);
    cdt.insertEdges(edges);
    cdt.eraseOuterTrianglesAndHoles();

    return cdt.triangles;
}

size_t countTriangles(const std::vector<CDT::TriangleVec>& triangulated_polygons)
{
    size_t total = 0;
    for(const auto& triangles : triangulated_polygons)
    {
        total += triangles.size();
    }
    return total;
}
