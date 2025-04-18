#include <torch/torch.h>
#include <torch/script.h>       // For loading the TorchScript surrogate model
#include <omp.h>                // OpenMP for parallel rollouts
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// Hyperparameters
struct PPOHyperparams {
    int    num_envs = 8;        // number of parallel environments (threads)
    int    horizon = 10;       // time steps per episode
    int    epochs = 3;        // number of PPO update epochs per batch
    float  gamma = 0.99f;    // discount factor
    float  lam = 0.95f;    // GAE lambda for advantage estimation
    float  clip_epsilon = 0.2f;     // PPO clip threshold
    float  lr = 3e-4f;    // learning rate for Adam
    float  value_coef = 0.5f;     // coefficient for value loss
    float  entropy_coef = 0.01f;    // coefficient for entropy bonus
    int    max_iterations = 1000;    // training iterations (batches)
    int    validate_interval = 100;  // how often to print/validate progress
};

// Environment for wing geometry optimization using the surrogate model
class WingEnv {
public:
    // Geometry state: [chord, max_thick, x_thick, max_camber, x_camber]
    std::array<float, 5> geom;
    float AoA;          // angle of attack (constant for now)
    float initial_ld;   // L/D at start of episode
    float last_ld;      // L/D after last action

    // Parameter bounds for realism/constraints (min and max for each param)
    std::array<float, 5> min_bounds = { 0.5f, 0.05f, 0.2f, 0.0f, 0.0f };  // example lower bounds
    std::array<float, 5> max_bounds = { 2.0f, 0.20f, 0.6f, 0.1f, 1.0f };  // example upper bounds

    // Pointer to surrogate model (shared across envs)
    torch::jit::script::Module* surrogate;

    WingEnv(torch::jit::script::Module* surrogate_model, float fixedAoA = 5.0f)
        : AoA(fixedAoA), surrogate(surrogate_model) {}

    // Reset the environment with a random initial geometry. Returns initial state.
    std::array<float, 6> reset() {
        // Randomize geometry within bounds (uniform)
        for (int i = 0; i < 5; ++i) {
            float minv = min_bounds[i];
            float maxv = max_bounds[i];
            // simple uniform random in [minv, maxv]
            float r = static_cast<float>(rand()) / RAND_MAX;
            geom[i] = minv + r * (maxv - minv);
        }
        // Compute initial L/D using surrogate
        initial_ld = querySurrogate(geom, AoA);
        last_ld = initial_ld;
        // Return state (geometry + AoA)
        std::array<float, 6> state;
        for (int i = 0; i < 5; ++i) state[i] = geom[i];
        state[5] = AoA;
        return state;
    }

    // Step the environment with the given action (parameter adjustments).
    // Returns (next_state, reward, done) - done is not used here since we handle episode length externally.
    std::tuple<std::array<float, 6>, float, bool> step(const std::array<float, 5>& action) {
        // Apply action to geometry and clamp to bounds
        for (int i = 0; i < 5; ++i) {
            geom[i] += action[i];  // adjust parameter
            // enforce bounds
            if (geom[i] < min_bounds[i]) geom[i] = min_bounds[i];
            if (geom[i] > max_bounds[i]) geom[i] = max_bounds[i];
        }
        // Query surrogate for new L/D
        float new_ld = querySurrogate(geom, AoA);
        // Calculate reward as improvement in L/D
        float reward = new_ld - last_ld;
        // Update last_ld
        last_ld = new_ld;
        // Construct next state (geometry + AoA)
        std::array<float, 6> next_state;
        for (int i = 0; i < 5; ++i) next_state[i] = geom[i];
        next_state[5] = AoA;
        // (In this setup, 'done' is determined by external episode length, so always false here)
        bool done = false;
        return { next_state, reward, done };
    }

private:
    // Helper to query the surrogate model (predict L/D for given geometry and AoA)
    float querySurrogate(const std::array<float, 5>& geom, float AoA) {
        // Create input tensor [1 x 6] (batch of 1)
        torch::Tensor input = torch::zeros({ 1, 6 }, torch::kFloat32);
        for (int i = 0; i < 5; ++i) {
            input[0][i] = geom[i];
        }
        input[0][5] = AoA;
        // Run surrogate model (no gradient needed, inference mode)
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        torch::Tensor output = surrogate->forward(inputs).toTensor();
        // Assume output is a single value (L/D)
        float ld = output.item<float>();
        return ld;
    }
};

// Actor-Critic Network (Policy and Value combined)
struct ActorCriticImpl : torch::nn::Module {
    // Layers
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
    torch::nn::Linear action_head{ nullptr };  // outputs action mean for 5 parameters
    torch::nn::Linear value_head{ nullptr };   // outputs state value
    // Log std dev for action distribution (as a learnable parameter, one per action dim)
    torch::Tensor log_std;
    int action_dim;

    ActorCriticImpl(int state_dim = 6, int action_dim = 5, int hidden_size = 64) : action_dim(action_dim) {
        // Initialize neural network layers
        fc1 = register_module("fc1", torch::nn::Linear(state_dim, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        action_head = register_module("action_head", torch::nn::Linear(hidden_size, action_dim));
        value_head = register_module("value_head", torch::nn::Linear(hidden_size, 1));
        // Initialize log_std as learnable parameters (start with 0 -> std=1 for each action dim)
        log_std = register_parameter("log_std", torch::zeros({ action_dim }));
    }

    // Forward pass: given state, return (action_mean, state_value)
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor state) {
        // Apply feedforward layers with ReLU activations
        torch::Tensor x = torch::relu(fc1->forward(state));
        x = torch::relu(fc2->forward(x));
        // Policy mean and value
        torch::Tensor action_mean = action_head->forward(x);   // shape: [batch, action_dim]
        torch::Tensor state_value = value_head->forward(x);    // shape: [batch, 1]
        return { action_mean, state_value };
    }
};
TORCH_MODULE(ActorCritic);  // Macro to create module holder for ActorCriticImpl

int main() {
    // Seed for reproducibility
    srand(time(NULL));

    // Load the surrogate model
    std::string model_path = "surrogate_model.pt";
    torch::jit::script::Module surrogate_model;
    try {
        surrogate_model = torch::jit::load(model_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the surrogate model from " << model_path << std::endl;
        return -1;
    }
    surrogate_model.eval();  // ensure surrogate is in eval mode

    // Hyperparameters
    PPOHyperparams cfg;

    // Initialize policy/value network
    ActorCritic model(/*state_dim=*/6, /*action_dim=*/5, /*hidden_size=*/64);
    model->train();  // set in training mode (though we have no dropout in this network)

    // Optimizer for both actor and critic parameters
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(cfg.lr));

    // Create parallel environments
    int M = cfg.num_envs;
    std::vector<WingEnv> envs;
    envs.reserve(M);
    for (int i = 0; i < M; ++i) {
        envs.emplace_back(&surrogate_model, 5.0f);  // AoA = 5 degrees fixed
    }

    // Variables for tracking progress
    double best_ld = -std::numeric_limits<double>::infinity();
    std::array<float, 5> best_geom;

    // Training loop
    for (int iter = 1; iter <= cfg.max_iterations; ++iter) {
        // Storage for rollout data
        int T = cfg.horizon;
        int batch_size = M * T;
        std::vector<float> states_buf(batch_size * 6);
        std::vector<float> actions_buf(batch_size * 5);
        std::vector<float> rewards_buf(batch_size);
        std::vector<float> values_buf(batch_size);
        std::vector<float> old_log_probs_buf(batch_size);
        std::vector<bool> done_buf(batch_size);

        // Run parallel rollouts
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            // Each thread will work on envs[i]
            int thread_id = omp_get_thread_num();
            WingEnv& env = envs[i];
            // Reset environment to a new random initial geometry
            std::array<float, 6> state = env.reset();
            // Compute initial state value (baseline) using value network
            // (We'll also get policy output to sample action)
            for (int t = 0; t < T; ++t) {
                // Flatten index for storage
                int idx = i * T + t;
                // Copy state into buffer
                for (int j = 0; j < 6; ++j) {
                    states_buf[idx * 6 + j] = state[j];
                }
                // Get policy mean and value for current state
                // (Wrap state into a tensor for network forward)
                torch::Tensor state_tensor = torch::from_blob(state.data(), { 1, 6 }).clone().to(torch::kFloat32);
                auto [action_mean_tensor, value_tensor] = model->forward(state_tensor);
                // Get action mean as 1x5 and value as 1x1
                torch::Tensor action_mean = action_mean_tensor.squeeze(0);
                torch::Tensor state_val = value_tensor.squeeze(0);
                // Sample action from Gaussian policy
                // (Use mean and std from model->log_std)
                torch::Tensor log_std = model->log_std;              // shape [5]
                torch::Tensor std = torch::exp(log_std);
                // Sample a random Gaussian vector for action
                torch::Tensor eps = torch::randn_like(action_mean);
                torch::Tensor action_tensor = action_mean + eps * std;
                // Compute log probability of this action under the policy
                torch::Tensor diff = (action_tensor - action_mean) / std;
                torch::Tensor log_prob_per_dim = -0.5 * diff.pow(2) - log_std - 0.5 * torch::log(2 * torch::tensor(M_PI));
                torch::Tensor log_prob = log_prob_per_dim.sum();  // scalar (for this state)

                // Extract action and log_prob values to CPU
                std::array<float, 5> action;
                for (int k = 0; k < 5; ++k) {
                    action[k] = action_tensor[k].item<float>();
                }
                float logp = log_prob.item<float>();
                float value = state_val.item<float>();

                // Store action, value, log_prob
                for (int k = 0; k < 5; ++k) {
                    actions_buf[idx * 5 + k] = action[k];
                }
                values_buf[idx] = value;
                old_log_probs_buf[idx] = logp;

                // Step environment with chosen action
                auto [next_state, reward, done] = env.step(action);
                rewards_buf[idx] = reward;
                done_buf[idx] = done;

                // Update state for next step
                state = next_state;
            }
        } // end parallel for

        // Compute advantages and returns for the collected batch
        std::vector<float> advantages_buf(batch_size);
        std::vector<float> returns_buf(batch_size);
        // Initialize advantages to 0
        std::fill(advantages_buf.begin(), advantages_buf.end(), 0.0f);

        // Calculate returns (discounted sum of rewards) and advantages with GAE
        for (int i = 0; i < M; ++i) {
            // for each episode in batch
            int base = i * T;
            float last_gae = 0.0f;
            float future_return = 0.0f;
            // We assume episode ends at T (done), so for t = T-1:
            // if the episode was truncated artificially, we could bootstrap V(next), but here done at end.
            // Set next_value = 0 for terminal
            for (int t = T - 1; t >= 0; --t) {
                int idx = base + t;
                // If not the last step, we can use next state's value; if last step, next_value = 0 (terminal)
                float next_value = 0.0f;
                if (t < T - 1) {
                    // not terminal step
                    next_value = values_buf[idx + 1];
                }
                // If an environment had a true early termination (done_buf true), we'd reset next_value=0, but here done all false until end.
                float delta = rewards_buf[idx] + cfg.gamma * ((t < T - 1) ? next_value : 0.0f) - values_buf[idx];
                // GAE advantage
                last_gae = delta + cfg.gamma * cfg.lam * last_gae;
                advantages_buf[idx] = last_gae;
                // Compute return (discounted reward to end of episode)
                future_return = rewards_buf[idx] + cfg.gamma * future_return;
                returns_buf[idx] = future_return;
            }
        }
        // Normalize advantages for numerical stability (zero mean, unit std)
        double adv_sum = 0.0, adv_sumsq = 0.0;
        for (float adv : advantages_buf) {
            adv_sum += adv;
            adv_sumsq += adv * adv;
        }
        double adv_mean = adv_sum / batch_size;
        double adv_var = adv_sumsq / batch_size - adv_mean * adv_mean;
        double adv_std = (adv_var > 1e-8) ? std::sqrt(adv_var) : 1e-8;
        for (float& adv : advantages_buf) {
            adv = static_cast<float>((adv - adv_mean) / adv_std);
        }

        // Convert buffers to torch Tensors for batch training
        torch::Tensor states = torch::from_blob(states_buf.data(), { batch_size, 6 }).clone();
        torch::Tensor actions = torch::from_blob(actions_buf.data(), { batch_size, 5 }).clone();
        torch::Tensor old_log_probs = torch::from_blob(old_log_probs_buf.data(), { batch_size }).clone();
        torch::Tensor returns = torch::from_blob(returns_buf.data(), { batch_size }).clone();
        torch::Tensor advantages = torch::from_blob(advantages_buf.data(), { batch_size }).clone();

        // PPO update: train for a few epochs on this batch
        for (int epoch = 0; epoch < cfg.epochs; ++epoch) {
            // We use the full batch in this example (could be divided into minibatches for larger batch sizes)
            // Forward pass for current policy
            auto [action_mean_batch, value_batch] = model->forward(states);
            // Compute new log probabilities of the taken actions under current policy
            // Expand log_std to match batch size
            torch::Tensor log_std = model->log_std;  // shape [5]
            // Expand to [batch_size, 5] for broadcasting
            torch::Tensor log_std_expanded = log_std.expand({ actions.size(0), log_std.size(0) });
            torch::Tensor std = torch::exp(log_std_expanded);
            torch::Tensor diff = (actions - action_mean_batch) / std;
            torch::Tensor log_prob_per_dim = -0.5 * diff.pow(2) - log_std_expanded - 0.5 * torch::log(2 * torch::tensor(M_PI));
            torch::Tensor log_probs_new = log_prob_per_dim.sum(1);  // sum over action dimensions -> shape [batch_size]

            // Ratio for clipping
            torch::Tensor ratio = torch::exp(log_probs_new - old_log_probs);
            // Advantage as tensor (ensure same device and shape)
            torch::Tensor adv = advantages;
            // Compute surrogate objectives
            torch::Tensor surrogate1 = ratio * adv;
            torch::Tensor surrogate2 = torch::clamp(ratio, 1.0f - cfg.clip_epsilon, 1.0f + cfg.clip_epsilon) * adv;
            torch::Tensor policy_loss = -torch::mean(torch::min(surrogate1, surrogate2));

            // Value loss (MSE between predicted value and return)
            torch::Tensor value_pred = value_batch.squeeze(1);
            torch::Tensor value_loss = torch::mean((value_pred - returns).pow(2));

            // Entropy bonus (encourage exploration)
            // Compute entropy of Gaussian policy: H = 0.5 * sum(1 + log(2πσ^2)) (per time-step)
            torch::Tensor entropy_per_dim = 0.5 * (1 + (2 * torch::tensor(M_PI) * std.pow(2)).log());
            torch::Tensor entropy = entropy_per_dim.sum(1).mean();  // mean entropy across batch
            torch::Tensor entropy_loss = -cfg.entropy_coef * entropy;

            // Total loss
            torch::Tensor total_loss = policy_loss + cfg.value_coef * value_loss + entropy_loss;

            // Update policy and value network
            optimizer.zero_grad();
            total_loss.backward();
            // (Optional: clip gradients for stability, e.g., torch::nn::utils::clip_grad_norm_(model.parameters(), max_norm))
            optimizer.step();
        }

        // (Optional) Track progress and print/log
        if (iter % cfg.validate_interval == 0 || iter == cfg.max_iterations) {
            // Compute average episode return (sum of rewards) and average final L/D across all envs
            double avg_return = 0.0;
            double avg_final_ld = 0.0;
            for (int i = 0; i < M; ++i) {
                // For each env episode, return = final L/D - initial L/D (telescopic sum of improvements)
                float episode_return = envs[i].last_ld - envs[i].initial_ld;
                avg_return += episode_return;
                avg_final_ld += envs[i].last_ld;
                // Update best design found
                if (envs[i].last_ld > best_ld) {
                    best_ld = envs[i].last_ld;
                    best_geom = envs[i].geom;  // copy best geometry parameters
                }
            }
            avg_return /= M;
            avg_final_ld /= M;
            std::cout << "Iteration " << iter
                << ": Avg Return = " << avg_return
                << ", Avg Final L/D = " << avg_final_ld
                << ", Best L/D so far = " << best_ld << std::endl;
        }
    }

    // After training, output the best geometry found
    std::cout << "Best wing design found has L/D = " << best_ld << " with parameters: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << best_geom[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // (Optional) You could now run a high-fidelity simulation (e.g., XFoil) on best_geom to validate the actual L/D.
    return 0;
}
