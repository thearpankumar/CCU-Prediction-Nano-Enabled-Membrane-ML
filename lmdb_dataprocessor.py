import lmdb
import os
import pickle
import pandas as pd
import numpy as np

# Define the base directory
base_dir = '/home/warmachine/codes/Projcts/Nano-Enabled-Membrane/dataset/is2r/train'

# Check if the directory exists
if not os.path.exists(base_dir):
    print(f"Error: Directory '{base_dir}' does not exist.")
    exit(1)

# List all .lmdb files in the directory
lmdb_files = [f for f in os.listdir(base_dir) if f.endswith('.lmdb')]
if not lmdb_files:
    print(f"Error: No .lmdb files found in '{base_dir}'.")
    exit(1)

# List to store all data entries
all_data = []

# Process each .lmdb file
for lmdb_file in lmdb_files:
    lmdb_path = os.path.join(base_dir, lmdb_file)
    print(f"Processing {lmdb_file}")
    try:
        env = lmdb.open(lmdb_path, readonly=True, subdir=False, lock=False)
        with env.begin() as txn:
            for key, value in txn.cursor():
                try:
                    key_str = key.decode()
                    data = pickle.loads(value)
                    data_entry = {'file': lmdb_file, 'key': key_str}
                    # Update with all fields from the data dictionary
                    data_entry.update(data)
                    # Convert pos_relaxed to string to save in CSV (or process later)
                    if 'pos_relaxed' in data_entry:
                        data_entry['pos_relaxed'] = str(data_entry['pos_relaxed'])
                    all_data.append(data_entry)
                except (pickle.UnpicklingError, KeyError) as e:
                    print(f"Error in {lmdb_file}, key {key_str}: {e}")
        env.close()
    except lmdb.Error as e:
        print(f"LMDB Error for {lmdb_file}: {e}")

# Create DataFrame
df = pd.DataFrame(all_data)
print("DataFrame columns:", df.columns.tolist())

# Save to CSV
csv_path = 'is2r_train_data.csv'
df.to_csv(csv_path, index=False)
print(f"Data saved to '{csv_path}'")