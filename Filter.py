import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Load the dataset
file_path = r'File_path'
data = pd.read_excel(file_path)

# Function to calculate the sample rate from the 'Time (s)' column
def calculate_sample_rate(time_column):
    time_diffs = np.diff(time_column)  # Differences between consecutive time values
    avg_time_step = np.mean(time_diffs)  # Average time step
    return 1 / avg_time_step  # Sample rate is the reciprocal of the average time step

# Function to create a Butterworth lowpass filter
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Calculate the sampling rate from the data
fs = calculate_sample_rate(data['Time (s)'])  # Dynamically determined sample rate
print(f"Calculated Sampling Rate: {fs} Hz")

# Parameters for the filter
order = 3
cutoff = 7  # Desired cutoff frequency, can be adjusted based on the signal

# Generate the filter coefficients
b, a = butter_lowpass(cutoff, fs, order)

# Create a new DataFrame to store filtered data
filtered_df = pd.DataFrame()
filtered_df['Time (s)'] = data['Time (s)']  # Copy the time column

# Apply filtering and plot for each column except 'Time (s)'
for column in data.columns:
    if column == 'Time (s)':
        continue
    
    filtered_data = filtfilt(b, a, data[column])
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['Time (s)'], data[column], label='Original')
    plt.plot(data['Time (s)'], filtered_data, label='Filtered', color = 'red')
    plt.xlabel('Time(s)')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.title(f'{column} Signal Filtering')
    plt.show()