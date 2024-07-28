import struct
import numpy as np

# Define the file path
file_path = './C2_1LGCC_1.bin'

# Read the binary data from the file
with open(file_path, 'rb') as file:
    data = file.read()

# Extract relevant information from the binary data according to the specifications in the PDF
# Assuming the data starts from address 0x1470
data_offset = 0x1470
analog_data = data[data_offset:]

# Convert the analog data to a numpy array of unsigned 8-bit integers
analog_values = np.frombuffer(analog_data, dtype=np.uint8)
#convert to float
analog_values = analog_values.astype(float)

# Define parameters for voltage conversion
ch_volt_div_val = 1000000  # Example value for V/div in mV
code_per_div = 256  # Example value for total data code in a horizontal division
ch_vert_offset = 0  # Example value for vertical offset in V

# Convert the data to voltage using the formula from the PDF
voltages = (analog_values - 128) * ch_volt_div_val / 1000 / code_per_div + ch_vert_offset

# Plot the data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(voltages, label='Channel 1')
plt.title('Analog Channel Data')
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.show()
