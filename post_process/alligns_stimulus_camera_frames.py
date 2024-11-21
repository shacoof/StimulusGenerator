import pandas as pd
from datetime import datetime
import numpy as np

# Convert timestamp strings to datetime objects for easy comparison
def parse_timestamp(ts):
    try:
        return datetime.strptime(ts, "%H:%M:%S:%f")
    except Exception:
        return None

# Define paths
output_path = r"Z:\Lab-Shared\Data\ClosedLoop\Dev Trails\20241113\20241113_162022_t9-fixed_angle\stimuli_log_w_frames.csv"
stimulus_log_path = r"Z:\Lab-Shared\Data\ClosedLoop\Dev Trails\20241113\20241113_162022_t9-fixed_angle\stimuli_log.csv"
camera_frames_path = r"Z:\Lab-Shared\Data\ClosedLoop\Dev Trails\20241113\20241113_162022_t9-fixed_angle\20241113_162022_t9-fixed_angle_log.csv"

# Create DataFrames
stimulus_log = pd.read_csv(stimulus_log_path)
camera_frames = pd.read_csv(camera_frames_path)

# Convert timestamps to datetime objects for easier processing
stimulus_log["TS_dt"] = stimulus_log["TS"].apply(parse_timestamp)
camera_frames["timestamp_dt"] = camera_frames["timestamp"].apply(parse_timestamp)

# Extract datetime values if the column contains tuples
if isinstance(camera_frames["timestamp_dt"].iloc[0], tuple):
    camera_frames["timestamp_dt"] = camera_frames["timestamp_dt"].apply(lambda x: x[1])

# Compute frame numbers
frame_numbers = []
for ts in stimulus_log["TS_dt"]:
    if ts is not None:
        # Calculate the closest timestamp
        time_differences = (camera_frames["timestamp_dt"] - ts).abs()
        closest_idx = time_differences.idxmin()
        frame_numbers.append(camera_frames.loc[closest_idx, "image no"])
    else:
        frame_numbers.append(None)

# Add the frame numbers to the first DataFrame
stimulus_log["frame number"] = frame_numbers

# Save to CSV
stimulus_log.to_csv(output_path, index=False)

print(stimulus_log)
