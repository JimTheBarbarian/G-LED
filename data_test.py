import numpy as np
import os

# Define the filename
filename = 'data1.npy'


# Load the data from the .npy file
try:
    data = np.load(filename, allow_pickle=True) # allow_pickle=True if it might contain objects
    print(f"Successfully loaded '{filename}'")
    #print("\nContents:")
    #print(data)

    # Check if the loaded data is a NumPy array and print its shape
    if isinstance(data, np.ndarray):
        print(f"\nShape of the array: {data.shape}")
    # Handle cases where the .npy file might contain other structures (like dictionaries saved with np.save)
    elif isinstance(data, dict):
        print("\nLoaded data is a dictionary.")
        for key, value in data.items():
            print(f"  Key: '{key}'")
            if isinstance(value, np.ndarray):
                print(f"    Value is a NumPy array with shape: {value.shape}")
            else:
                print(f"    Value type: {type(value)}")
    elif hasattr(data, 'shape'): # Check if it has a shape attribute even if not strictly ndarray
         print(f"\nShape of the loaded object: {data.shape}")
    else:
        print(f"\nLoaded data is of type: {type(data)}")
        print("Cannot determine shape for this data type.")

except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print(f"An error occurred while loading or processing the file: {e}")
