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

std::pair<CDT::TriangleVec, int> triangulatePolygon(const PolygonWithHoles& poly)
{
    // Insert all vertices (outer + holes)
    std::vector<CDT::V2d<float>> all_vertices = poly.outer;
    std::vector<size_t> outer_indices, hole_start_indices;
    for(size_t i = 0; i < poly.outer.size(); ++i) outer_indices.push_back(i);
    for(const auto& hole : poly.holes) {
        hole_start_indices.push_back(all_vertices.size());
        all_vertices.insert(all_vertices.end(), hole.begin(), hole.end());
    }
    
    // Remove duplicate vertices - this handles touching points correctly
    auto duplicates_info = CDT::RemoveDuplicates(all_vertices);
    
    // Insert outer ring edges (adjusted for removed duplicates)
    std::vector<CDT::Edge> edges;
    size_t original_outer_size = poly.outer.size();
    for(size_t i = 0; i < original_outer_size; ++i) {
        size_t next = (i + 1) % original_outer_size;
        // Only add edge if the vertices are different after duplicate removal
        size_t mapped_i = duplicates_info.mapping[i];
        size_t mapped_next = duplicates_info.mapping[next];
        if (mapped_i != mapped_next) {
            edges.push_back(CDT::Edge(CDT::VertInd(mapped_i), CDT::VertInd(mapped_next)));
        }
    }
    
    // Insert hole edges (adjusted for removed duplicates)
    size_t vert_offset = original_outer_size;
    for(const auto& hole : poly.holes) {
        for(size_t i = 0; i < hole.size(); ++i) {
            size_t idx1 = vert_offset + i;
            size_t idx2 = vert_offset + ((i + 1) % hole.size());
            // Only add edge if the vertices are different after duplicate removal
            size_t mapped_idx1 = duplicates_info.mapping[idx1];
            size_t mapped_idx2 = duplicates_info.mapping[idx2];
            if (mapped_idx1 != mapped_idx2) {
                edges.push_back(CDT::Edge(CDT::VertInd(mapped_idx1), CDT::VertInd(mapped_idx2)));
            }
        }
        vert_offset += hole.size();
    }
    
    // Try different CDT strategies
    try {
        // Method 0: Default CDT
        CDT::Triangulation<float> cdt;
        cdt.insertVertices(all_vertices);
        cdt.insertEdges(edges);
        cdt.eraseOuterTrianglesAndHoles();
        return std::make_pair(cdt.triangles, 0);
    } catch (const CDT::IntersectingConstraintsError&) {
        try {
            // Method 1: Try to resolve intersections
            CDT::Triangulation<float> cdt_resolve(
                CDT::VertexInsertionOrder::AsProvided,
                CDT::IntersectingConstraintEdges::TryResolve,
                0.0f
            );
            cdt_resolve.insertVertices(all_vertices);
            cdt_resolve.insertEdges(edges);
            cdt_resolve.eraseOuterTrianglesAndHoles();
            return std::make_pair(cdt_resolve.triangles, 1);
        } catch (const std::exception&) {
            try {
                // Method 2: Don't check for intersections
                CDT::Triangulation<float> cdt_nocheck(
                    CDT::VertexInsertionOrder::AsProvided,
                    CDT::IntersectingConstraintEdges::DontCheck,
                    0.0f
                );
                cdt_nocheck.insertVertices(all_vertices);
                cdt_nocheck.insertEdges(edges);
                cdt_nocheck.eraseOuterTrianglesAndHoles();
                return std::make_pair(cdt_nocheck.triangles, 2);
            } catch (const std::exception&) {
                // Method 3: All methods failed
                return std::make_pair(CDT::TriangleVec(), 3);
            }
        }
    } catch (const std::exception&) {
        // Method 3: All methods failed
        return std::make_pair(CDT::TriangleVec(), 3);
    }
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
