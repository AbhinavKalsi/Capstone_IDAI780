// predict_surrogate.cpp
// Compile exactly like train_surrogate, but source file is predict_surrogate.cpp

#include <torch/torch.h>
#include <iostream>
#include <array>

// Must match the architecture in train_surrogate.cpp
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

int main(int argc, char* argv[]) {
    // 1) Load the trained surrogate
    auto model = std::make_shared<SurrogateNet>();
    torch::load(model, "surrogate_model.pt");
    model->eval();

    // 2) Build a feature vector: [AoA, chord, max_thick, x_thick, max_cam, avg_xtr]
    //    Here we pick an example; in PPO you'll construct this from your current geometry.
    std::array<float, 6> feat = {
        /* aoa     */ 5.0f,
        /* chord   */ 1.0f,
        /* max_thick */ 0.08f,
        /* x_thick   */ 0.25f,
        /* max_cam   */ 0.04f,
        /* avg_xtr   */ 0.50f
    };

    // 3) Convert to a 1×6 tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor x = torch::from_blob(feat.data(), { 1,6 }, options).clone();

    // 4) Predict
    torch::Tensor y = model->forward(x);
    float ld_pred = y.item<float>();
    std::cout << "Predicted L/D = " << ld_pred << std::endl;

    return 0;
}
