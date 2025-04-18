import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import joblib

# Load surrogate model and scalers
model_surrogate = SurrogateNet()
model_surrogate.load_state_dict(torch.load('surrogate_model.pth'))
model_surrogate.eval()

scaler_x = joblib.load('scaler_x.save')
scaler_y = joblib.load('scaler_y.save')

# Load PPO policy
model_ppo = PPO.load("ppo_airfoil_model")

# Create environment
env = AirfoilEnv()

obs, _ = env.reset()
for step in range(10):
    action, _states = model_ppo.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: Action={action}, Predicted L/D={info['predicted_LD']}")

print("PPO policy evaluation complete.")
