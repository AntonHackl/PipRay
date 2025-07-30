// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "triangulation.h"
#include <fstream>
#include <iostream>

void extractPolygonWithHoles(const boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<float>>& poly,
                             std::vector<PolygonWithHoles>& out)
{
    PolygonWithHoles pwh;
    auto outer = boost::geometry::exterior_ring(poly);
    for(const auto& pt : outer) {
        pwh.outer.push_back({pt.x(), pt.y()});
    }
    if(!pwh.outer.empty() && pwh.outer.front().x == pwh.outer.back().x &&
       pwh.outer.front().y == pwh.outer.back().y) {
        pwh.outer.pop_back();
    }
    if (pwh.outer.size() < 3) {
        std::cerr << "Polygon outer ring has less than 3 vertices after processing. Skipping polygon..." << std::endl;
        return;
    }
    size_t num_holes = boost::geometry::num_interior_rings(poly);
    const auto& holes = boost::geometry::interior_rings(poly);
    for(size_t h = 0; h < num_holes; ++h) {
        std::vector<CDT::V2d<float>> hole_vertices;
        try {
            const auto& hole = holes[h];
            if (hole.empty() || hole.size() < 3) {
                continue;
            }
            for(const auto& pt : hole) {
                hole_vertices.push_back({pt.x(), pt.y()});
            }
            if(!hole_vertices.empty() && hole_vertices.front().x == hole_vertices.back().x &&
               hole_vertices.front().y == hole_vertices.back().y) {
                hole_vertices.pop_back();
            }
            pwh.holes.push_back(hole_vertices);
        } catch(const std::exception& e) {
            std::cerr << "Error accessing polygon hole: " << e.what() << std::endl;
            continue;
        }
    }
    out.push_back(pwh);
}


std::vector<PolygonWithHoles> readPolygonVerticesFromFile(const std::string& filepath)
{
    using BoostPolygon = boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<float>>;
    using BoostMultiPolygon = boost::geometry::model::multi_polygon<BoostPolygon>;
    std::ifstream file(filepath);
    std::vector<std::string> lines;
    std::string line;
    while(getline(file, line))
    {
        // Find lines containing POLYGON or MULTIPOLYGON
        if((line.find("POLYGON") != std::string::npos || line.find("MULTIPOLYGON") != std::string::npos))
        {
            lines.push_back(line);
        }
    }
    if(lines.empty())
    {
        return {};
    }
    std::vector<PolygonWithHoles> all_polygons;
    for(const auto& wkt : lines)
    {
        try {
            if(wkt.find("MULTIPOLYGON") != std::string::npos) {
                BoostMultiPolygon multipoly;
                boost::geometry::read_wkt(wkt, multipoly);
                for(const auto& poly : multipoly) {
                    extractPolygonWithHoles(poly, all_polygons);
                }
            } else if(wkt.find("POLYGON") != std::string::npos) {
                BoostPolygon poly;
                boost::geometry::read_wkt(wkt, poly);
                extractPolygonWithHoles(poly, all_polygons);
            }
        } catch(const std::exception& e) {
            std::cerr << "Error reading WKT: " << e.what() << std::endl;
        }
    }
    return all_polygons;
}

CDT::TriangleVec triangulatePolygon(const PolygonWithHoles& poly)
{
    CDT::Triangulation<float> cdt;

    // Insert all vertices (outer + holes)
    std::vector<CDT::V2d<float>> all_vertices = poly.outer;
    std::vector<size_t> outer_indices, hole_start_indices;
    for(size_t i = 0; i < poly.outer.size(); ++i) outer_indices.push_back(i);
    for(const auto& hole : poly.holes) {
        hole_start_indices.push_back(all_vertices.size());
        all_vertices.insert(all_vertices.end(), hole.begin(), hole.end());
    }
    cdt.insertVertices(all_vertices);

    // Insert outer ring edges
    std::vector<CDT::Edge> edges;
    for(size_t i = 0; i < poly.outer.size(); ++i) {
        size_t next = (i + 1) % poly.outer.size();
        edges.push_back(CDT::Edge(CDT::VertInd(i), CDT::VertInd(next)));
    }
    // Insert hole edges
    size_t vert_offset = poly.outer.size();
    for(const auto& hole : poly.holes) {
        for(size_t i = 0; i < hole.size(); ++i) {
            size_t idx1 = vert_offset + i;
            size_t idx2 = vert_offset + ((i + 1) % hole.size());
            edges.push_back(CDT::Edge(CDT::VertInd(idx1), CDT::VertInd(idx2)));
        }
        vert_offset += hole.size();
    }
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
