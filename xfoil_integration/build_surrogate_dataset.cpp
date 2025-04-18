// build_surrogate_dataset.cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;
using Row = std::vector<double>;

// 1) Compute geometry features from (x,y) pairs
Row computeFeatures(const std::vector<std::pair<double, double>>& coords) {
    int N = coords.size();
    double chord = 1.0;                    // normalized
    int mid = N / 2;
    double max_thick = 0, x_at_thick = 0;
    double max_camber = 0, x_at_camber = 0;
    for (int i = 1; i < mid; i++) {
        double yu = coords[i].second;
        double yl = coords[N - 1 - i].second;
        double thick = yu - yl;
        if (thick > max_thick) {
            max_thick = thick;
            x_at_thick = coords[i].first;
        }
        double cam = 0.5 * (yu + yl);
        if (std::abs(cam) > std::abs(max_camber)) {
            max_camber = cam;
            x_at_camber = coords[i].first;
        }
    }
    return { chord, max_thick, x_at_thick, max_camber, x_at_camber };
}

int main() {
    // --- 1) Read and parse final_cfd_results.csv ---
    std::ifstream in("final_cfd_results.csv");
    if (!in) { std::cerr << "Error: cannot open final_cfd_results.csv\n"; return 1; }
    std::string line;
    std::getline(in, line); // skip header

    struct Perf {
        std::string foil;
        double aoa, CL, CD, CDp, CM, Top, Bot, LD;
    };
    std::vector<Perf> perf;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        Perf p; std::string tok;

        std::getline(ss, p.foil, ',');
        std::getline(ss, tok, ','); p.aoa = std::stod(tok);
        std::getline(ss, tok, ','); p.CL = std::stod(tok);
        std::getline(ss, tok, ','); p.CD = std::stod(tok);
        std::getline(ss, tok, ','); p.CDp = std::stod(tok);
        std::getline(ss, tok, ','); p.CM = std::stod(tok);
        std::getline(ss, tok, ','); p.Top = std::stod(tok);
        std::getline(ss, tok, ','); p.Bot = std::stod(tok);
        std::getline(ss, tok, ','); p.LD = std::stod(tok);

        perf.push_back(p);
    }
    in.close();

    // --- 2) For each unique foil, load its .csv and compute features ---
    std::map<std::string, Row> feats_map;
    fs::path base = fs::current_path() / "Dataset" / "parsed_data";

    for (auto& p : perf) {
        if (feats_map.count(p.foil)) continue;              // already done

        fs::path csvf = base / (p.foil + ".csv");
        if (!fs::exists(csvf)) {
            std::cerr << "Warning: missing geometry file " << csvf << "\n";
            continue;
        }

        std::ifstream fcsv(csvf.string());
        if (!fcsv) {
            std::cerr << "Warning: cannot open " << csvf << "\n";
            continue;
        }

        std::getline(fcsv, line); // skip header
        std::vector<std::pair<double, double>> coords;
        while (std::getline(fcsv, line)) {
            if (line.empty()) continue;
            std::stringstream ls(line);
            double x, y; char comma;
            if (!(ls >> x >> comma >> y)) continue;
            coords.emplace_back(x, y);
        }
        fcsv.close();

        if (coords.empty()) {
            std::cerr << "Warning: no coords in " << csvf << "\n";
            continue;
        }

        feats_map[p.foil] = computeFeatures(coords);
    }

    // --- 3) Write out surrogate_dataset.csv only for foils with features ---
    std::ofstream out("surrogate_dataset.csv");
    out << "foil,aoa,CL,CD,LD,"
        "chord,max_thick,x_thick,max_camber,x_camber,avg_xtr\n";

    for (auto& p : perf) {
        auto it = feats_map.find(p.foil);
        if (it == feats_map.end()) continue;  // skip missing
        auto& g = it->second;
        double avg_xtr = 0.5 * (p.Top + p.Bot);
        out << p.foil << "," << p.aoa << "," << p.CL << "," << p.CD << "," << p.LD
            << "," << g[0] << "," << g[1] << "," << g[2] << "," << g[3] << "," << g[4]
            << "," << avg_xtr << "\n";
    }
    out.close();
    std::cout << "Wrote surrogate_dataset.csv with " << feats_map.size() << " entries\n";
    return 0;
}



