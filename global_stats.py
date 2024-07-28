import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures

# Define the folder path containing the .bin files
folder_path = './'

# Parameters for voltage conversion
data_offset = 0x1470
ch_volt_div_val = 1000000  # Example value for V/div in mV
code_per_div = 256  # Example value for total data code in a horizontal division
ch_vert_offset = 0  # Example value for vertical offset in V

# Function to process a single .bin file and extract voltage data
def process_file(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8)
    analog_values = analog_values.astype(float)
    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    
    # Compute summary statistics
    summary = {
        'File Name': os.path.basename(file_path),
        'Mean (V)': np.mean(voltages),
        'Median (V)': np.median(voltages),
        'Min (V)': np.min(voltages),
        'Max (V)': np.max(voltages),
        'Std Dev (V)': np.std(voltages)
    }
    
    return summary

# Get list of .bin files in the folder
file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.bin')]

# Process files concurrently
summaries = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
    for future in concurrent.futures.as_completed(future_to_file):
        try:
            summaries.append(future.result())
        except Exception as exc:
            print(f'File {future_to_file[future]} generated an exception: {exc}')

# Create a summary dataframe
summary_df = pd.DataFrame(summaries)

# Optionally, save the summary dataframe to a CSV file
summary_df.to_csv('summary_statistics.csv', index=False)

# Plotting example for one of the files
if file_paths:
    plt.figure(figsize=(10, 6))
    voltages = process_file(file_paths[0])
    plt.plot(voltages, label=os.path.basename(file_paths[0]))
    plt.title('Analog Channel Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Voltage (V)')
