import os
import struct
import numpy as np

# Define the function to convert binary file to numpy array
def convert_bin_to_npy(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()

    data_offset = 0x1470
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8)
    analog_values = analog_values.astype(float)

    ch_volt_div_val = 5000  # Example value for V/div in mV
    code_per_div = 25  # Example value for total data code in a horizontal division
    ch_vert_offset = -7.7  # Example value for vertical offset in V

    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    return voltages

# Define the function to process all .bin files in a directory
def process_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                print(f'Processing file: {file_path}')
                voltages = convert_bin_to_npy(file_path)
                npy_file_path = file_path.replace('.bin', '.npy')
                np.save(npy_file_path, voltages)
                print(f'Saved numpy array to: {npy_file_path}')

# Define the directory path to be processed
directory_path = './your_directory_path_here'

# Process the directory
process_directory(directory_path)
