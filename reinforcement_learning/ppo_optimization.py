import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import joblib

# SurrogateNet class definition (exact same as previously)
class SurrogateNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Updated AirfoilEnv (with normalization)
class AirfoilEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.model = SurrogateNet()
        self.model.load_state_dict(torch.load('surrogate_model.pth'))
        self.model.eval()

        self.scaler_x = joblib.load('scaler_x.save')
        self.scaler_y = joblib.load('scaler_y.save')

        self.state = np.array([5.0, 1.0, 0.08, 0.25, 0.04, 0.50], dtype=np.float32)
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([5.0, 1.0, 0.08, 0.25, 0.04, 0.50], dtype=np.float32)
        normalized_state = self.scaler_x.transform([self.state])[0]
        return normalized_state, {}

    def step(self, action):
        self.state = np.clip(self.state + action, -10.0, 10.0)
        
        input_scaled = torch.FloatTensor(self.scaler_x.transform([self.state]))
        with torch.no_grad():
            predicted_ld_scaled = self.model(input_scaled).numpy()
            predicted_ld = self.scaler_y.inverse_transform(predicted_ld_scaled)[0, 0]

        reward = (predicted_ld - 50) / 20.0  # Normalized Reward
        
        terminated = False
        truncated = False
        info = {'predicted_LD': predicted_ld}

        normalized_state = self.scaler_x.transform([self.state])[0]

        return normalized_state, reward, terminated, truncated, info

# Run PPO optimization again
env = make_vec_env(lambda: AirfoilEnv(), n_envs=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("ppo_airfoil_model_normalized")

print("PPO training complete and normalized model saved.")
