import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_airfoil_dat(filename):
    """
    Reads a UIUC airfoil .dat file and returns an array of coordinates.
    It skips header lines and any comments (lines starting with '#').
    """
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Loop over lines and skip headers/comments.
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()
        # Skip empty lines and comment lines
        if not line or line.startswith('#'):
            continue
        # If the first token has letters, it's likely a header line.
        first_token = line.split()[0]
        if any(c.isalpha() for c in first_token):
            continue
        # Try converting the first two tokens into floats (x and y)
        try:
            x, y = float(line.split()[0]), float(line.split()[1])
            data.append([x, y])
        except (IndexError, ValueError):
            continue
    return np.array(data)

# Example usage:
filename = 'e61.dat'  # Replace with your actual file name.
airfoil_coords = read_airfoil_dat(filename)

# Optional: Save the processed data to CSV for future use.
df = pd.DataFrame(airfoil_coords, columns=['x', 'y'])
df.to_csv('NACA0012.csv', index=False)

# Visualize the airfoil
plt.figure(figsize=(8, 4))
plt.plot(airfoil_coords[:, 0], airfoil_coords[:, 1], marker='o', markersize=3, linestyle='-')
plt.title('Airfoil Profile from UIUC Dataset')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.axis('equal')
plt.grid(True)
plt.show()
