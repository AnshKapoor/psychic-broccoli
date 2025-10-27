import pandas as pd
import numpy as np
import logging
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# from Flight_Trajectory import find_trajectory, load_all_data

'''Run Data_Preparation.py first'''
# project_path = os.path.abspath('..')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def detect_anomalies(data: np.ndarray, window_size: int = 5, std_threshold: float = 3.0) -> np.ndarray:
    """
    Detect anomalous points using statistics

    Args:
        data (np.ArrayLike): Input numerical array
        window_size (int): Size of rolling window
        std_threshold (float): Number of standard deviations of outlier detection

    Returns:
        np.ArrayLike: Boolean mask where True indicates normal points, False indicates anomalies
    """
    rolling_mean = pd.Series(data).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(data).rolling(window=window_size, center=True).std()

    lower_bound = rolling_mean - (std_threshold * rolling_std)
    upper_bound = rolling_mean + (std_threshold * rolling_std)

    return (data >= lower_bound) & (data <= upper_bound)


def preprocess_trajectory(trajectory_df: pd.DataFrame, target_length: int = 20) -> pd.DataFrame:
    """
    Apply regularized reconstruction to smooth trajectory and remove anomalies

    Args:
        trajectory_df (pd.DataFrame): Input trajectory DataFrame
        target_length (int): Target length of trajectory

    Returns:
         pd.DataFrame: Cleaned and smoothed trajectory DataFrame
    """

    numerical_cols = ['latitude', 'longitude', 'groundspeed', 'track', 'altitude']
    df_cleaned = trajectory_df.copy().reset_index(drop=True)

    # Create a combined mask for all columns
    all_valid = np.ones(len(df_cleaned), dtype=bool)

    '1. Remove anomalies'
    for col in numerical_cols:

        if col in df_cleaned.columns:
            # Detect anomalies and remove anomalies
            col_valid = detect_anomalies(df_cleaned[col].values)
            # Update combined mask - only keep rows that pass all checks
            all_valid = np.logical_and(all_valid, col_valid)

    # Apply the combined mask once
    df_cleaned = df_cleaned[all_valid].reset_index(drop=True)

    # If trajectory is too short, return None
    if len(trajectory_df) < 3:  # PCHIP needs at least 3 points
        return pd.DataFrame()

    '2. Apply Savgol-Filter for smoothing if enough points remain'
    for col in numerical_cols:
        if col in df_cleaned.columns:
            window_length = min(11, len(df_cleaned) - 1)
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                df_cleaned[col] = savgol_filter(df_cleaned[col], window_length, polyorder=3)

    '3. Interpolate numerical columns using PCHIP'
    orig_time = np.linspace(0, 1, len(df_cleaned))
    new_time = np.linspace(0, 1, target_length)

    result_data = {}

    # Interpolate numerical columns
    for col in numerical_cols:
        if col in df_cleaned.columns:
            pchip = PchipInterpolator(orig_time, df_cleaned[col].values)
            result_data[col] = pchip(new_time)

    # Handle non-numerical columns with nearest neighbor
    categorical_cols = ['icao24', 'callsign']
    for col in categorical_cols:
        if col in df_cleaned.columns:
            result_data[col] = df_cleaned[col].iloc[0]

    # Handle timestamp specifically
    if 'timestamp' in df_cleaned.columns:

        try:
            # Convert timestamps to unix timestamps (seconds since epoch)
            if pd.api.types.is_datetime64_any_dtype(df_cleaned['timestamp']):
                # if timestamp are already datetime objects
                unix_timestamps = df_cleaned['timestamp'].astype('int64') // 10 ** 9
            else:
                unix_timestamps = pd.to_datetime(df_cleaned['timestamp']).astype('int64') // 10 ** 9

            # Interpolate unix timestamps
            pchip = PchipInterpolator(orig_time, unix_timestamps.values)
            interpolated_timestamps = pchip(new_time)

            # Convert back to datetime objects
            result_data['timestamp'] = pd.to_datetime(interpolated_timestamps, unit='s')

        except Exception as e:
            logging.error(f'Error interpolating: {e}')

            # Fallback: spread the first and last timestamp evenly
            if len(df_cleaned) > 0:
                first_ts = pd.to_datetime(df_cleaned['timestamp'].iloc[0])
                last_ts = pd.to_datetime(df_cleaned['timestamp'].iloc[-1])
                total_seconds = (last_ts - first_ts).total_seconds()

                # Create evenly spaced timestamps
                result_data['timestamp'] = [
                    first_ts + pd.Timedelta(second=i * total_seconds / (target_length - 1))
                    for i in range(target_length)
                ]
    # Create the DataFrame from result_data
    result_df = pd.DataFrame(result_data)

    # Reorder columns to put timestamp first if it exists
    if 'timestamp' in result_df.columns:
        # Get all columns
        all_columns = list(result_df.columns)
        # Remove timestamp from the list
        all_columns.remove('timestamp')
        # Create new column order with timestamp first
        new_column_order = ['timestamp'] + all_columns
        # Reorder the DataFrame
        result_df = result_df[new_column_order]

    return result_df
