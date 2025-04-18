import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

class SurrogateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# Load the model state dict only
model = SurrogateNet()
model.load_state_dict(torch.load('surrogate_model.pth'))
model.eval()

# Load scalers using joblib
scaler_x = joblib.load('scaler_x.save')
scaler_y = joblib.load('scaler_y.save')

# Example input:
features = np.array([[5.0, 1.0, 0.08, 0.25, 0.04, 0.50]])
features_scaled = torch.FloatTensor(scaler_x.transform(features))

with torch.no_grad():
    pred_scaled = model(features_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.numpy())
    print(f'Predicted L/D = {pred[0][0]:.2f}')
