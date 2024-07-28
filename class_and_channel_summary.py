import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures

# Define the folder path containing the .bin files
folder_path = './bin_files'

# Parameters for voltage conversion
data_offset = 0x1470
ch_volt_div_val = 1000000  # Example value for V/div in mV
code_per_div = 256  # Example value for total data code in a horizontal division
ch_vert_offset = 0  # Example value for vertical offset in V

# Class descriptions based on file naming
class_descriptions = {
    '2CC': '2 Covered Conductors',
    '3CC': '3 Covered Conductors',
    '1LGS': '1 Line, Ground, Steel Pole',
    '2LLS': '2 Lines, Steel Pole',
    '2LGS': '2 Lines, Ground, Steel Pole',
    '3LGS': '3 Lines, Ground, Steel Pole',
    '3LLS': '3 Lines, Steel Pole',
    '2LLCC': '2 Lines, Covered Conductor',
    '3LLCC': '3 Lines, Covered Conductor',
    '2LGCC': '2 Lines, Ground, Covered Conductor',
    '3LGCC': '3 Lines, Ground, Covered Conductor',
    '1LGCC': '1 Line, Ground, Covered Conductor',
    'BG': 'Background (no power)',
    'BG2': 'Background 2 (no power)',
    'BG3': 'Background 3 (no power)',
    'BGHV': 'Background, High Voltage',
    'BGHV2': 'Background, High Voltage 2'
}

# Function to process a single .bin file and extract voltage data
def process_file(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    
    analog_data = data[data_offset:]
    analog_values = np.frombuffer(analog_data, dtype=np.uint8)
    analog_values = analog_values.astype(float)
    voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset
    
    # Extract channel and type from filename
    parts = os.path.basename(file_path).split('_')
    channel_type = parts[1]
    sample_no = parts[2].split('.')[0]
    
    # Compute summary statistics
    summary = {
        'File Name': os.path.basename(file_path),
        'Channel_Type': channel_type,
        'Sample No': sample_no,
        'Mean (V)': np.mean(voltages),
        'Median (V)': np.median(voltages),
        'Min (V)': np.min(voltages),
        'Max (V)': np.max(voltages),
        'Std Dev (V)': np.std(voltages)
    }
    
    return summary, voltages

# Get list of .bin files in the folder
file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.bin')]

# Process files concurrently
summaries = []
voltages_by_group = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
    for future in concurrent.futures.as_completed(future_to_file):
        try:
            summary, voltages = future.result()
            summaries.append(summary)
            group_key = summary['Channel_Type']
            if group_key not in voltages_by_group:
                voltages_by_group[group_key] = []
            voltages_by_group[group_key].append(voltages)
        except Exception as exc:
            print(f'File {future_to_file[future]} generated an exception: {exc}')

# Create a summary dataframe
summary_df = pd.DataFrame(summaries)

# Display the summary dataframe
import ace_tools as tools; tools.display_dataframe_to_user(name="Summary Statistics", dataframe=summary_df)

# Optionally, save the summary dataframe to a CSV file
summary_df.to_csv('summary_statistics.csv', index=False)

# Plotting example for each group
for group_key, voltages_list in voltages_by_group.items():
    if voltages_list:
        plt.figure(figsize=(10, 6))
        for voltages in voltages_list:
            plt.plot(voltages, alpha=0.5)
        plt.title(f'Analog Channel Data for {group_key}')
        plt.xlabel('Sample Index')
        plt.ylabel('Voltage (V)')
        plt.legend([f'{group_key} {i+1}' for i in range(len(voltages_list))])
        plt.grid(True)
        plt.show()
