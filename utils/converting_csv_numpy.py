import os
import numpy as np
import pandas as pd

def read_airfoil_dat(filename):
    """
    Reads a UIUC airfoil .dat file and returns an array of coordinates.
    It skips header lines and any comments (lines starting with '#').
    """
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Process each line: skip headers/comments and convert to floats
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # If the first token contains alphabetic characters, skip (header)
        first_token = line.split()[0]
        if any(c.isalpha() for c in first_token):
            continue
        try:
            x, y = float(line.split()[0]), float(line.split()[1])
            data.append([x, y])
        except (IndexError, ValueError):
            continue
    return np.array(data)

# Create a new folder 'parsed_data' if it doesn't exist
output_folder = "parsed_data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all .dat files in the current directory
dat_files = [f for f in os.listdir('.') if f.lower().endswith('.dat')]

# Process each .dat file
for dat_file in dat_files:
    # Parse the airfoil data into a NumPy array
    coords = read_airfoil_dat(dat_file)
    
    if coords.size == 0:
        print(f"Warning: No coordinate data found in {dat_file}. Skipping.")
        continue

    # Save as CSV file
    df = pd.DataFrame(coords, columns=['x', 'y'])
    csv_filename = os.path.join(output_folder, os.path.splitext(dat_file)[0] + '.csv')
    df.to_csv(csv_filename, index=False)
    
    # Optionally, save the numpy array as .npy file
    npy_filename = os.path.join(output_folder, os.path.splitext(dat_file)[0] + '.npy')
    np.save(npy_filename, coords)
    
    print(f"Parsed {dat_file} -> Saved CSV: {csv_filename} and NPY: {npy_filename}")
