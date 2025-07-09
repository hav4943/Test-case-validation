import pandas as pd
import matplotlib.pyplot as plt

# Load the .xlsx dataset
file_path = r'file_path'  # Replace with the actual file path
data = pd.read_excel(file_path)

# Extract time and velocity columns
time = data['Time_1']  # Replace 'Time' with the actual column name for time
velocity = data['Vehicle_speed(km/h)']  # Replace 'Velocity' with the actual column name for velocity

# Calculate the original sampling rate
time_duration = time.iloc[-1] - time.iloc[0]  # Total time duration in seconds
initial_sampling_rate = len(time) / time_duration

# Downsampling factor
N = 5

# Downsample the data
downsampled_time = time.iloc[::N].reset_index(drop=True)
downsampled_velocity = velocity.iloc[::N].reset_index(drop=True)

# Calculate the new sampling rate
new_sampling_rate = initial_sampling_rate / N

# Plot the data before and after downsampling
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot original data
axes[0].plot(time, velocity, color='blue', label='Original Data')
axes[0].set_title(f"Original Data (Sampling Frequency: {initial_sampling_rate:.2f} Hz)", fontsize=14)
axes[0].set_xlabel('Time (s)', fontsize=12)
axes[0].set_ylabel('Velocity', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Plot downsampled data
axes[1].plot(downsampled_time, downsampled_velocity, color='red', label='Downsampled Data')
axes[1].set_title(f"Downsampled Data (Sampling Frequency: {new_sampling_rate:.2f} Hz)", fontsize=14)
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Velocity', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

      
