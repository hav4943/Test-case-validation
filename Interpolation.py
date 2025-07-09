import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = r"C:\Users\UZEX5M4\Test_case_Data\Signal_data\Resample& interpolation.xlsx"  # Replace with the actual file path
data = pd.read_excel(file_path)

# Extract columns
time_1 = data['Time_1']
value_1 = data['Vehicle_speed(km/h)']
time_2 = data['Time_2']
value_2 = data['FL_Wheel_speed(Km/h)']

# Define the time range for the subset
start_time = 95
end_time = start_time + 1  # Adjust this window as needed

# Create masks for the subsets
subset_mask_1 = (time_1 >= start_time) & (time_1 <= end_time)
subset_mask_2 = (time_2 >= start_time) & (time_2 <= end_time)

# Extract subsets
time_1_subset = time_1[subset_mask_1]
value_1_subset = value_1[subset_mask_1]
time_2_subset = time_2[subset_mask_2]
value_2_subset = value_2[subset_mask_2]

# Calculate original sampling frequencies
if len(time_1_subset) > 1:
    original_vehicle_speed_freq = len(time_1_subset) / (time_1_subset.max() - time_1_subset.min())
else:
    original_vehicle_speed_freq = 0

if len(time_2_subset) > 1:
    original_wheel_speed_freq = len(time_2_subset) / (time_2_subset.max() - time_2_subset.min())
else:
    original_wheel_speed_freq = 0

# Debug: Print extracted subsets to check data
print("Time_1 subset:", time_1_subset)
print("Value_1 subset:", value_1_subset)
print("Time_2 subset:", time_2_subset)
print("Value_2 subset:", value_2_subset)

# Define a common time grid for resampling within the subset
common_time_subset = np.linspace(
    max(time_1_subset.min(), time_2_subset.min()),
    min(time_1_subset.max(), time_2_subset.max()),
    num=100
)

# Ensure there are sufficient points for interpolation
if len(time_1_subset) > 1:
    value_1_resampled_subset = np.interp(common_time_subset, time_1_subset, value_1_subset)
    vehicle_speed_sampling_freq = len(common_time_subset) / (common_time_subset[-1] - common_time_subset[0])
else:
    print("Error: Insufficient points for velocity interpolation.")
    value_1_resampled_subset = np.zeros_like(common_time_subset)
    vehicle_speed_sampling_freq = 0

if len(time_2_subset) > 1:
    value_2_resampled_subset = np.interp(common_time_subset, time_2_subset, value_2_subset)
    wheel_speed_sampling_freq = len(common_time_subset) / (common_time_subset[-1] - common_time_subset[0])
else:
    print("Error: Insufficient points for wheel speed interpolation.")
    value_2_resampled_subset = np.zeros_like(common_time_subset)
    wheel_speed_sampling_freq = 0

# Create a 4-panel plot
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 20))
title_fontsize = 16
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12

# Panel 1: Original Points for Vehicle Speed in Subset
axes[0].scatter(time_1_subset, value_1_subset, color='blue', label='Original Vehicle Speed Data (Subset)', s=10)
axes[0].set_title(f"Original Vehicle Speed Data (Sampling Frequency: {original_vehicle_speed_freq:.2f} Hz)", fontsize=title_fontsize)
axes[0].set_xlabel('Time(s)', fontsize=label_fontsize)
axes[0].set_ylabel('Vehicle Speed (km/h)', fontsize=label_fontsize)
axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axes[0].legend(fontsize=legend_fontsize)
axes[0].grid(alpha=0.3)

# Panel 2: Original Points for Wheel Speed in Subset
axes[1].scatter(time_2_subset, value_2_subset, color='purple', label='Original FL Wheel Speed Data (Subset)', s=10)
axes[1].set_title(f"Original FL Wheel Speed Data (Sampling Frequency: {original_wheel_speed_freq:.2f} Hz)", fontsize=title_fontsize)
axes[1].set_xlabel('Time(s)', fontsize=label_fontsize)
axes[1].set_ylabel('FL Wheel Speed (km/h)', fontsize=label_fontsize)
axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axes[1].legend(fontsize=legend_fontsize)
axes[1].grid(alpha=0.3)

# Panel 3: Resampled and Interpolated Vehicle Speed
axes[2].scatter(common_time_subset, value_1_resampled_subset, color='green', label='Resampled Vehicle Speed', s=10, alpha=0.7)
axes[2].plot(common_time_subset, value_1_resampled_subset, color='red', label='Interpolated Vehicle Speed', alpha=0.7)
axes[2].set_title(f"Resampled and Interpolated Vehicle Speed Data (Sampling Frequency: {vehicle_speed_sampling_freq:.2f} Hz)", fontsize=title_fontsize)
axes[2].set_xlabel('Time(s)', fontsize=label_fontsize)
axes[2].set_ylabel('Vehicle Speed (km/h)', fontsize=label_fontsize)
axes[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axes[2].legend(fontsize=legend_fontsize)
axes[2].grid(alpha=0.3)

# Panel 4: Resampled and Interpolated FL Wheel Speed
axes[3].scatter(common_time_subset, value_2_resampled_subset, color='orange', label='Resampled FL Wheel Speed', s=10, alpha=0.7)
axes[3].plot(common_time_subset, value_2_resampled_subset, color='blue', label='Interpolated FL Wheel Speed', alpha=0.7)
axes[3].set_title(f"Resampled and Interpolated FL Wheel Speed Data (Sampling Frequency: {wheel_speed_sampling_freq:.2f} Hz)", fontsize=title_fontsize)
axes[3].set_xlabel('Time(s)', fontsize=label_fontsize)
axes[3].set_ylabel('FL Wheel Speed (km/h)', fontsize=label_fontsize)
axes[3].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axes[3].legend(fontsize=legend_fontsize)
axes[3].grid(alpha=0.3)

# Adjust layout and show plot
plt.tight_layout()
plt.show()

