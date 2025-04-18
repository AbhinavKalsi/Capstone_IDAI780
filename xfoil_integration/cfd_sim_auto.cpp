#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>
#include <omp.h>

namespace fs = std::filesystem;
using namespace std;

// Function to generate the XFoil input string for a given airfoil file, AoA, and specified output filenames.
string generateXFoilInput(const string& airfoilFile, double aoa, const string& polarSave, const string& polarDump) {
    ostringstream oss;
    oss << "LOAD " << airfoilFile << "\n"         // Load the airfoil file
        << "OPER\n"                               // Enter operating mode
        << "VISC 1000000\n"                       // Set Reynolds number to 1e6
        << "PACC\n"                               // Start polar accumulation (no filenames on this line)
        << polarSave << "\n"                      // Supply polar save filename
        << polarDump << "\n"                      // Supply polar dump filename
        << "ALFA " << aoa << "\n\n"                // Set angle-of-attack (with extra newline)
        << "QUIT\n\n";                           // Quit XFoil (with final newline)
    return oss.str();
}

int main() {
    // Define paths
    string datasetDir = "cleaned_data";                    // Folder containing .dat files
    string outputDir = "synthetic_cfd_outputs";       // Folder for XFoil output files
    fs::create_directories(outputDir);                // Create if not exists

    // Define simulation parameters
    vector<double> aoaValues(21, 0.0);                 
    for (int aoa = -5; aoa <= 15; ++aoa)
        aoaValues.push_back(static_cast<double>(aoa));
    double Re = 1e6;                                  // Reynolds number

    // Find all .dat files in datasetDir
    vector<fs::path> datFiles;
    for (const auto& entry : fs::directory_iterator(datasetDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".dat") {
            datFiles.push_back(entry.path());
        }
    }

    cout << "Found " << datFiles.size() << " .dat files in " << datasetDir << endl;

    // Process each airfoil file in parallel using OpenMP
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < datFiles.size(); i++) {
        fs::path airfoilPath = datFiles[i];
        string airfoilFile = airfoilPath.string();
        // Extract airfoil base name without extension
        string airfoilName = airfoilPath.stem().string();

        // Loop over each AoA value for this airfoil
        for (double aoa : aoaValues) {
            // Define output prefix (use forward slashes or double backslashes)
            ostringstream outPrefixOSS;
            outPrefixOSS << outputDir << "/" << airfoilName << "_aoa_" << aoa;
            string outputPrefix = outPrefixOSS.str();
            // Define full filenames for polar save and dump
            string polarSave = outputPrefix + ".pol";
            string polarDump = outputPrefix + ".dump";

            // Generate XFoil input commands
            string xfoilInput = generateXFoilInput(airfoilFile, aoa, polarSave, polarDump);

            // Write input commands to a temporary file (unique per thread using airfoil name and AoA)
            string inputFilename = outputDir + "/" + airfoilName + "_aoa_" + to_string(aoa) + "_input.txt";
            ofstream inputFile(inputFilename);
            if (!inputFile) {
#pragma omp critical
                cerr << "Error: Unable to create input file " << inputFilename << endl;
                continue;
            }
            inputFile << xfoilInput;
            inputFile.close();

            // Construct command to run XFoil (ensure xfoil.exe is in PATH)
            string command = "xfoil < " + inputFilename;
#pragma omp critical
            {
                cout << "Processing " << airfoilName << " at AoA " << aoa << "�..." << endl;
            }
            // Execute XFoil using system() call
            int ret = system(command.c_str());
            if (ret != 0) {
#pragma omp critical
                cerr << "XFoil error for " << airfoilName << " at AoA " << aoa << ". Return code: " << ret << endl;
            }
            else {
#pragma omp critical
                cout << "XFoil completed for " << airfoilName << " at AoA " << aoa << ". Output saved to " << polarSave << endl;
            }
        } // end AoA loop
    } // end file loop

    cout << "All XFoil simulations completed." << endl;
    return 0;
}
