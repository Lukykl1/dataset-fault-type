import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import concurrent.futures
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis

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
    
    # Extract fault type from filename
    parts = os.path.basename(file_path).split('_')
    fault_type = parts[1]
    fault_type_description = class_descriptions.get(fault_type, 'Unknown')
    
    return fault_type_description, voltages

# Get list of .bin files in the folder
file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path) if file_name.endswith('.bin')]

# Function to compute the FFT and extract spectral features
def compute_fft(voltages, sampling_rate=1):
    n = len(voltages)
    fft_values = fft(voltages)
    freqs = fftfreq(n, d=1/sampling_rate)
    
    # Only take the positive half of the spectrum
    idx = np.arange(0, n//2)
    freqs = freqs[idx]
    fft_values = np.abs(fft_values[idx])
    
    return freqs, fft_values

# Function to extract additional features
def extract_additional_features(voltages):
    num_peaks, _ = find_peaks(voltages)
    rms_value = np.sqrt(np.mean(np.square(voltages)))
    skewness = skew(voltages)
    kurt = kurtosis(voltages)
    
    features = {
        'Number of Peaks (Original)': len(num_peaks),
        'RMS Value': rms_value,
        'Skewness': skewness,
        'Kurtosis': kurt
    }
    return features

# Function to extract spectral features
def extract_spectral_features(freqs, fft_values, voltages, sampling_rate=1):
    dominant_freq = freqs[np.argmax(fft_values)]
    dominant_magnitude = np.max(fft_values)
    peaks, _ = find_peaks(fft_values)
    num_peaks = len(peaks)
    
    # Compute Power Spectral Density (PSD)
    freqs_psd, psd_values = welch(voltages, fs=sampling_rate, nperseg=1024)
    
    spectral_features = {
        'Dominant Frequency (Hz)': dominant_freq,
        'Dominant Magnitude': dominant_magnitude,
        'Number of Peaks (FFT)': num_peaks,
        'PSD Frequencies': freqs_psd,
        'PSD Values': psd_values
    }
    return spectral_features

# Process files concurrently
spectral_summaries = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
    for future in concurrent.futures.as_completed(future_to_file):
        try:
            fault_type, voltages = future.result()
            freqs, fft_values = compute_fft(voltages)
            spectral_features = extract_spectral_features(freqs, fft_values, voltages)
            additional_features = extract_additional_features(voltages)
            spectral_features.update(additional_features)
            spectral_features['File Name'] = os.path.basename(future_to_file[future])
            spectral_features['Fault Type'] = fault_type
            spectral_summaries.append(spectral_features)
        except Exception as exc:
            print(f'File {future_to_file[future]} generated an exception: {exc}')

# Create a summary dataframe for spectral features
spectral_summary_df = pd.DataFrame(spectral_summaries)

# Optionally, save the spectral summary dataframe to a CSV file
spectral_summary_df.to_csv('spectral_summary_statistics.csv', index=False)

# Spectral analysis and plotting for each fault type
for fault_type, group in spectral_summary_df.groupby('Fault Type'):
    plt.figure(figsize=(10, 6))
    for i, row in group.iterrows():
        file_path = os.path.join(folder_path, row['File Name'])
        _, voltages = process_file(file_path)
        freqs, fft_values = compute_fft(voltages)
        plt.plot(freqs, fft_values, alpha=0.5)
    plt.title(f'Spectral Analysis (FFT) for {fault_type}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend(group['File Name'])
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, row in group.iterrows():
        plt.plot(row['PSD Frequencies'], row['PSD Values'], alpha=0.5)
    plt.title(f'Spectral Analysis (PSD) for {fault_type}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend(group['File Name'])
    plt.grid(True)
    plt.show()
