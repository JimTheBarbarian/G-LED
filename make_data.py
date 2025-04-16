import torch
import numpy as np
import os

# Define the folder containing the tensor files
tensor_folder = '.' # Assuming tensors are in the current directory

# Define the number of tensors
num_tensors = 5000

# List to hold the loaded tensors
tensor_list = []

# Loop through the tensor files and load them
print("Loading tensors...")
for i in range(num_tensors):
    file_path = os.path.join(tensor_folder, f'traj{i}.pth')
    if os.path.exists(file_path):
        try:
            tensor = torch.load(file_path)
            stacked_traj_tensor = torch.stack([t.cpu() for t in tensor], dim=0)
            tensor_list.append(stacked_traj_tensor) # Append the [3841, 64] tensor
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Handle error appropriately, e.g., skip or exit
            exit()
    else:
        print(f"Warning: File not found {file_path}")
        # Handle missing file appropriately, e.g., skip or exit
        exit()

    # Optional: Print progress
    if (i + 1) % 100 == 0:
        print(f"Loaded {i + 1}/{num_tensors} tensors.")

# Check if we loaded the expected number of tensors
if len(tensor_list) != num_tensors:
    print(f"Error: Expected {num_tensors} tensors, but loaded {len(tensor_list)}.")
    exit()

# Stack the tensors into a single tensor
print("Stacking tensors...")
try:
    combined_tensor = torch.stack(tensor_list, dim=0)
except Exception as e:
    print(f"Error stacking tensors: {e}")
    # Check shapes if stacking fails
    for idx, t in enumerate(tensor_list):
        if t.shape != tensor_list[0].shape:
            print(f"Shape mismatch at index {idx}: expected {tensor_list[0].shape}, got {t.shape}")
    exit()


print(f"Combined tensor shape: {combined_tensor.shape}")

# Convert the combined tensor to a NumPy array
print("Converting to NumPy array...")
numpy_array = combined_tensor.numpy()

# Save the NumPy array to a file
output_file = 'data0.npy'
print(f"Saving NumPy array to {output_file}...")
np.save(output_file, numpy_array)

print("Done.")