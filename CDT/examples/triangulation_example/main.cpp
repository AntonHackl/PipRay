#include <CDT.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/io/wkt/wkt.hpp>

using namespace std;

void printUsage(const string& programName)
{
    cerr << "Usage: " << programName << " -i <input_file> -o <output_file>"
         << endl;
    cerr << "Options:" << endl;
    cerr << "  -i <input_file>    Input WKT file containing polygons" << endl;
    cerr << "  -o <output_file>   Output WKT file for triangulated triangles"
         << endl;
    cerr << "  -h, --help         Show this help message" << endl;
    cerr << endl;
    cerr << "Example: " << programName << " -i input.wkt -o triangles.wkt"
         << endl;
}

// Helper function to read the first POLYGON entry from a file and extract
// vertices using Boost.Geometry from a WKT file
vector<vector<CDT::V2d<float> > >
readPolygonVerticesFromFile(const string& filepath)
{
    using BoostPolygon = boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<float> >;
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
    vector<vector<CDT::V2d<float> > > polygon_vertices;
    // Only use the exterior ring
    for(const auto& poly : polygons)
    {
        vector<CDT::V2d<float> > vertices;
        if(!boost::geometry::is_valid(poly))
        {
            cerr << "Invalid polygon found in WKT." << endl;
            continue;
        }
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
        polygon_vertices.push_back(vertices);
    }
    return polygon_vertices;
}

CDT::TriangleVec triangulatePolygon(const vector<CDT::V2d<float> >& polygon)
{
    CDT::Triangulation<float> cdt;

    // Copy polygon to allow modification by RemoveDuplicates
    vector<CDT::V2d<float> > vertices = polygon;
    // Remove duplicates in-place
    auto duplicates_info = CDT::RemoveDuplicates(vertices);

    // Create edges
    vector<CDT::Edge> edges(vertices.size(), CDT::Edge(0, 0));
    for(size_t i = 0; i < vertices.size(); ++i)
    {
        edges[i] = {i, (i + 1) % vertices.size()};
    }

    // Insert vertices
    cdt.insertVertices(vertices);

    // Insert boundary edges
    cdt.insertEdges(edges);

    // Perform the triangulation
    cdt.eraseOuterTrianglesAndHoles();

    return cdt.triangles;
}

int main(int argc, char* argv[])
{
    string inputFilepath;
    string outputFilepath;

    // Parse command line arguments
    for(int i = 1; i < argc; ++i)
    {
        string arg = argv[i];

        if(arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if((arg == "-i" || arg == "--input") && i + 1 < argc)
        {
            inputFilepath = argv[++i];
        }
        else if((arg == "-o" || arg == "--output") && i + 1 < argc)
        {
            outputFilepath = argv[++i];
        }
        else
        {
            cerr << "Unknown argument: " << arg << endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Check if required arguments are provided
    if(inputFilepath.empty() || outputFilepath.empty())
    {
        cerr << "Error: Both input file (-i) and output file (-o) must be "
                "specified."
             << endl;
        printUsage(argv[0]);
        return 1;
    }

    // Check if input file exists
    ifstream testFile(inputFilepath);
    if(!testFile.good())
    {
        cerr << "Error: Input file '" << inputFilepath
             << "' does not exist or cannot be opened." << endl;
        return 1;
    }
    testFile.close();

    vector<vector<CDT::V2d<float> > > polygons =
        readPolygonVerticesFromFile(inputFilepath);

    if(polygons.empty())
    {
        cerr << "Error: No valid polygons found in input file '"
             << inputFilepath << "'." << endl;
        return 1;
    }

    vector<CDT::TriangleVec> triangulated_polygons;
    for(const auto& poly : polygons)
    {
        auto triangulated = triangulatePolygon(poly);
        triangulated_polygons.push_back(triangulated);
    }

    // Print the results
    // cout << "Number of polygons: " << polygons.size() << endl;

    // Save triangles as WKT polygons, grouped by original polygon
    ofstream wkt_out(outputFilepath);
    if(!wkt_out.is_open())
    {
        cerr << "Error: Cannot open output file '" << outputFilepath
             << "' for writing." << endl;
        return 1;
    }

    using BoostPolygon = boost::geometry::model::polygon<
        boost::geometry::model::d2::point_xy<float> >;
    for(size_t poly_idx = 0; poly_idx < triangulated_polygons.size();
        ++poly_idx)
    {
        const auto& triangles = triangulated_polygons[poly_idx];
        const auto& vertices = polygons[poly_idx];
        // Write a separator or comment for each polygon group
        wkt_out << "# Triangles for polygon group " << poly_idx << std::endl;
        for(const auto& tri : triangles)
        {
            BoostPolygon poly;
            poly.outer().push_back(
                {vertices[tri.vertices[0]].x, vertices[tri.vertices[0]].y});
            poly.outer().push_back(
                {vertices[tri.vertices[1]].x, vertices[tri.vertices[1]].y});
            poly.outer().push_back(
                {vertices[tri.vertices[2]].x, vertices[tri.vertices[2]].y});
            // Close the ring by repeating the first point
            poly.outer().push_back(
                {vertices[tri.vertices[0]].x, vertices[tri.vertices[0]].y});
            wkt_out << boost::geometry::wkt(poly) << std::endl;
        }
        wkt_out << std::endl; // Blank line between groups
    }
    wkt_out.close();

    cout << "Triangulation completed successfully!" << endl;
    cout << "Input file: " << inputFilepath << endl;
    cout << "Output file: " << outputFilepath << endl;
    cout << "Number of polygons processed: " << polygons.size() << endl;

    return 0;
}