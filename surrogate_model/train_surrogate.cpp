// train_surrogate.cpp
// Compile with: 
// g++ -std=c++17 train_surrogate.cpp -o train_surrogate \
//     -I /path/to/libtorch/include -I /path/to/libtorch/include/torch/csrc/api/include \
//     -L /path/to/libtorch/lib -ltorch -lc10 -Wl,-rpath,/path/to/libtorch/lib

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// A tiny MLP: 6 inputs → 64 → 64 → 1 output
struct SurrogateNet : torch::nn::Module {
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
    SurrogateNet() {
        fc1 = register_module("fc1", torch::nn::Linear(6, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 64));
        fc3 = register_module("fc3", torch::nn::Linear(64, 1));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

int main() {
    // 1) Load CSV into host vectors
    std::ifstream in("surrogate_dataset.csv");
    if (!in) {
        std::cerr << "Error: cannot open surrogate_dataset.csv\n";
        return 1;
    }
    std::string line;
    std::getline(in, line); // skip header

    std::vector<std::array<float, 6>> feats;
    std::vector<float> targets;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;

        // Columns: foil,aoa,CL,CD,LD,chord,max_thick,x_thick,max_camber,x_camber,avg_xtr
        std::getline(ss, tok, ',');             // foil (ignored)
        std::getline(ss, tok, ','); float aoa = std::stof(tok);
        std::getline(ss, tok, ','); float CL = std::stof(tok);
        std::getline(ss, tok, ','); float CD = std::stof(tok);
        std::getline(ss, tok, ','); float LD = std::stof(tok);
        std::getline(ss, tok, ','); float chord = std::stof(tok);
        std::getline(ss, tok, ','); float max_thick = std::stof(tok);
        std::getline(ss, tok, ','); float x_thick = std::stof(tok);
        std::getline(ss, tok, ','); float max_cam = std::stof(tok);
        std::getline(ss, tok, ','); float x_cam = std::stof(tok);
        std::getline(ss, tok, ','); float avg_xtr = std::stof(tok);

        feats.push_back({ aoa, chord, max_thick, x_thick, max_cam, avg_xtr });
        targets.push_back(LD);
    }
    in.close();
    size_t N = feats.size();
    std::cout << "Loaded " << N << " samples\n";

    // 2) Convert to Torch tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor X = torch::empty({ (long)N,6 }, options);
    torch::Tensor Y = torch::empty({ (long)N,1 }, options);
    for (size_t i = 0; i < N; ++i) {
        X[i] = torch::from_blob(feats[i].data(), { 6 }, options).clone();
        Y[i][0] = targets[i];
    }

    // 3) Build model & optimizer
    auto model = std::make_shared<SurrogateNet>();
    model->train();
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    // 4) Training loop
    const int epochs = 500;
    for (int e = 1; e <= epochs; ++e) {
        optimizer.zero_grad();
        auto pred = model->forward(X);
        auto loss = torch::mse_loss(pred, Y);
        loss.backward();
        optimizer.step();
        if (e % 50 == 0) {
            std::cout << "Epoch " << e << "/" << epochs
                << "  Loss=" << loss.item<float>() << "\n";
        }
    }

    // 5) Save the trained model
    torch::save(model, "surrogate_model.pt");
    std::cout << "Surrogate model saved to surrogate_model.pt\n";
    return 0;
}
