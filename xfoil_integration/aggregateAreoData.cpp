#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <limits>
#include <omp.h>

namespace fs = std::filesystem;
using namespace std;

struct AeroData {
    string airfoil;
    double aoa;
    double CL;
    double CD;
    double CDp;
    double CM;
    double Top_Xtr;
    double Bot_Xtr;
    double LD;
};

bool parsePolarFile(const string& filepath, AeroData& data, const string& airfoilName) {
    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error: Could not open " << filepath << endl;
        return false;
    }

    string line;
    bool dataStarted = false;
    vector<AeroData> validEntries;

    while (getline(file, line)) {
        if (!dataStarted) {
            // Look for the dashed line indicating start of data
            if (line.find("------") != string::npos) {
                dataStarted = true;
            }
            continue;
        }

        // Process data lines
        istringstream iss(line);
        vector<double> values;
        string token;
        bool validLine = true;

        // Split line into numeric values
        while (iss >> token) {
            try {
                values.push_back(stod(token));
            }
            catch (...) {
                validLine = false;
                break;
            }
        }

        // Check for exactly 7 values (alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr)
        if (validLine && values.size() == 7) {
            AeroData entry;
            entry.airfoil = airfoilName;
            entry.aoa = values[0];
            entry.CL = values[1];
            entry.CD = values[2];
            entry.CDp = values[3];
            entry.CM = values[4];
            entry.Top_Xtr = values[5];
            entry.Bot_Xtr = values[6];
            entry.LD = (entry.CD != 0.0) ? (entry.CL / entry.CD) : 0.0;
            validEntries.push_back(entry);
        }
    }

    // Use the last valid entry if available
    if (!validEntries.empty()) {
        data = validEntries.back();
        return true;
    }

    return false;
}

int main() {
    string outputDir = "synthetic_cfd_outputs";
    string aggregateCSV = "final_cfd_results.csv";

    // Collect all polar files
    vector<fs::path> polarFiles;
    for (const auto& entry : fs::directory_iterator(outputDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".pol") {
            polarFiles.push_back(entry.path());
        }
    }

    cout << "Found " << polarFiles.size() << " polar files in " << outputDir << endl;

    vector<AeroData> allResults;

    // Process files in parallel
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < polarFiles.size(); i++) {
        fs::path pFile = polarFiles[i];
        string filepath = pFile.string();
        string filename = pFile.stem().string();

        // Extract airfoil name from filename
        size_t pos = filename.find("_aoa_");
        if (pos == string::npos) {
#pragma omp critical
            cerr << "Warning: Skipping improperly named file " << filepath << endl;
            continue;
        }
        string airfoilName = filename.substr(0, pos);

        AeroData data;
        if (parsePolarFile(filepath, data, airfoilName)) {
#pragma omp critical
            {
                allResults.push_back(data);
                cout << "Success: " << filepath
                    << " (AoA = " << data.aoa
                    << ", CL = " << data.CL
                    << ")" << endl;
            }
        }
        else {
#pragma omp critical
            {
                cerr << "Warning: No valid data in " << filepath
                    << " (non-converged)" << endl;
            }
        }
    }

    // Write results to CSV
    ofstream csvFile(aggregateCSV);
    if (!csvFile.is_open()) {
        cerr << "Error: Unable to create " << aggregateCSV << endl;
        return 1;
    }
    csvFile << "Airfoil,AoA,CL,CD,CDp,CM,Top_Xtr,Bot_Xtr,L/D\n";
    for (const auto& r : allResults) {
        csvFile << r.airfoil << ","
            << r.aoa << ","
            << r.CL << ","
            << r.CD << ","
            << r.CDp << ","
            << r.CM << ","
            << r.Top_Xtr << ","
            << r.Bot_Xtr << ","
            << r.LD << "\n";
    }
    csvFile.close();

    cout << "\nAggregated " << allResults.size() << " results in " << aggregateCSV << endl;

    // Sanity check statistics
    if (!allResults.empty()) {
        double min_CL = numeric_limits<double>::max();
        double max_CL = numeric_limits<double>::lowest();
        double sum_CL = 0.0;

        double min_CD = numeric_limits<double>::max();
        double max_CD = numeric_limits<double>::lowest();
        double sum_CD = 0.0;

        for (const auto& r : allResults) {
            min_CL = min(min_CL, r.CL);
            max_CL = max(max_CL, r.CL);
            sum_CL += r.CL;

            min_CD = min(min_CD, r.CD);
            max_CD = max(max_CD, r.CD);
            sum_CD += r.CD;
        }

        size_t n = allResults.size();
        cout << "\n--- Data Validation ---\n"
            << "CL range: " << min_CL << " to " << max_CL
            << " (avg: " << (sum_CL / n) << ")\n"
            << "CD range: " << min_CD << " to " << max_CD
            << " (avg: " << (sum_CD / n) << ")\n";
    }
    else {
        cout << "No valid results - check input files and convergence" << endl;
    }

    return 0;
}