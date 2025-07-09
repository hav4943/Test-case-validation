import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming df has a column 'Value' with your time series data
file_path = 'C:\\Users\\UZEX5M4\\Test_case_Data\\Signal_data\\Test_case_1\\Test_case_1.xlsx'
df = pd.read_excel(file_path)
time_series = df['Time'].dropna()

# Apply FFT
fft_values = np.fft.fft(time_series)
fft_freqs = np.fft.fftfreq(len(time_series), d=1)  # d=1 assuming uniform time steps

# Compute the magnitude of the FFT coefficients
magnitude = np.abs(fft_values)

# Select only the positive frequencies
positive_freqs = fft_freqs[:len(fft_freqs)//2]
positive_magnitude = magnitude[:len(magnitude)//2]

# Plot the frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, positive_magnitude)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
