import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('surrogate_dataset.csv')

X = data[['aoa', 'chord', 'max_thick', 'x_thick', 'max_camber', 'avg_xtr']].values
y = data['LD'].values.reshape(-1, 1)

scaler_x = StandardScaler().fit(X)
scaler_y = StandardScaler().fit(y)

X_train, X_test, y_train, y_test = train_test_split(
    scaler_x.transform(X), scaler_y.transform(y), test_size=0.2, random_state=42)

X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

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

model = SurrogateNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss={loss.item()}')

test_loss = criterion(model(X_test), y_test)
print(f'Test Loss={test_loss.item()}')

# Save model state dict only
torch.save(model.state_dict(), 'surrogate_model.pth')

# Save scalers separately using joblib
joblib.dump(scaler_x, 'scaler_x.save')
joblib.dump(scaler_y, 'scaler_y.save')
