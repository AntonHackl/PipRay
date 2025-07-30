// Prevent Windows.h from defining min/max macros that conflict with std::min/max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "triangulation.h"

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

int main(int argc, char* argv[])
{
    string inputFilepath;
    string outputFilepath;

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

    if(inputFilepath.empty() || outputFilepath.empty())
    {
        cerr << "Error: Both input file (-i) and output file (-o) must be "
                "specified."
             << endl;
        printUsage(argv[0]);
        return 1;
    }

    ifstream testFile(inputFilepath);
    if(!testFile.good())
    {
        cerr << "Error: Input file '" << inputFilepath
             << "' does not exist or cannot be opened." << endl;
        return 1;
    }
    testFile.close();

    // Read polygons from the input WKT file using triangulation library
    vector<PolygonWithHoles> polygons;
    try {
        polygons = readPolygonVerticesFromFile(inputFilepath);
    } catch (const exception& e) {
        cerr << "Error reading input file: " << e.what() << endl;
        return 1;
    }
    
    if(polygons.empty())
    {
        cerr << "Error: No valid polygons found in input file '"
             << inputFilepath << "'." << endl;
        return 1;
    }

    // Triangulate all polygons using the triangulation library
    vector<CDT::TriangleVec> triangulated_polygons;
    for (const auto& polygon : polygons) {
        try {
            CDT::TriangleVec triangles = triangulatePolygon(polygon);
            triangulated_polygons.push_back(triangles);
        } catch (const exception& e) {
            cerr << "Error triangulating polygon: " << e.what() << endl;
            return 1;
        }
    }

    // cout << "Number of polygons: " << polygons.size() << endl;

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
        wkt_out << "# Triangles for polygon " << poly_idx << std::endl;
        for(const auto& tri : triangles)
        {
            BoostPolygon poly;
            poly.outer().push_back(
                {vertices[tri.vertices[0]].x, vertices[tri.vertices[0]].y});
            poly.outer().push_back(
                {vertices[tri.vertices[1]].x, vertices[tri.vertices[1]].y});
            poly.outer().push_back(
                {vertices[tri.vertices[2]].x, vertices[tri.vertices[2]].y});
            poly.outer().push_back(
                {vertices[tri.vertices[0]].x, vertices[tri.vertices[0]].y});
            wkt_out << boost::geometry::wkt(poly) << std::endl;
        }
        wkt_out << std::endl;
    }
    wkt_out.close();

    cout << "Triangulation completed successfully!" << endl;
    cout << "Input file: " << inputFilepath << endl;
    cout << "Output file: " << outputFilepath << endl;
    cout << "Number of polygons processed: " << polygons.size() << endl;

    return 0;
}