import numpy as np
import pandas as pd
import logging
import os
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
project_path = os.path.abspath('..')


def normalize_timestamps(weather_file_1: str, weather_file_2: str, bp_results_file: str) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
    """
    Normalize timestamps of best practice data and weather data

    Args:
        weather_file_1 (str): Path to weather data contains: temperature, atmospheric pressure, relative humidity
        weather_file_2 (str):Path to weather data contains: wind speed, direction
        bp_results_file (str): Path to best practice results

    Returns:
        Tuple containing:
        - bp_data: Best practice data
        - w_data_2022: Weather information since 2022
    """

    # Read the weather data
    df1 = pd.read_csv(weather_file_1, delimiter=';')
    df2 = pd.read_csv(weather_file_2, delimiter=';')

    # Combine weather datasets
    weather_data = pd.concat([df1, df2], axis=1)
    weather_data = weather_data.loc[:, ~weather_data.columns.duplicated()]

    # Read best practice results
    best_practice_result = pd.read_csv(bp_results_file, delimiter=';')

    # ----------------------------------------------------------------
    # Normalize timestamps of best practice data
    # ----------------------------------------------------------------
    bp_data = best_practice_result.copy()
    timestamps_col = 'ATA/ATD'
    bp_data[timestamps_col] = bp_data[timestamps_col].astype(str)
    bp_data[timestamps_col] = pd.to_datetime(bp_data[timestamps_col], format='%d.%m.%Y %H:%M',
                                             errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Sort the best practice result by measurement points and ATD/ATA
    bp_data = bp_data.sort_values(by=["MP", "ATA/ATD"])

    # ----------------------------------------------------------------
    # Normalize timestamps of weather data
    # ----------------------------------------------------------------
    w_data = weather_data.copy()
    timestamps_col = 'MESS_DATUM'
    w_data[timestamps_col] = pd.to_numeric(w_data[timestamps_col], errors='coerce')
    w_data[timestamps_col] = pd.to_datetime(w_data[timestamps_col], format='%Y%m%d%H%M',
                                            errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Filtering to retain only 2022 data
    w_data[timestamps_col] = pd.to_datetime(w_data[timestamps_col], errors='coerce')
    w_data_2022 = w_data[w_data[timestamps_col].dt.year == 2022].reset_index(drop=True)

    return bp_data, w_data_2022


def combine_noise_data_sheets(base_path: str, excel_filename: str, csv_filename: str):
    """
    Merge noise data sheets of measurement points from EXCEL file into one CSV file.

    Args:
        base_path (str): Path of Data Folder
        excel_filename (str): Filepath of EXCEL file
        csv_filename (str): Filepath of CSV file
    """
    # logging.info('Starting to combine noise data sheets...')

    # Create a blank list to store all data
    all_data = []

    try:
        noise_data = os.path.join(base_path, excel_filename)
        noise_data_xls = pd.ExcelFile(noise_data)

        # Iterate all the sheets
        for sheet_name in noise_data_xls.sheet_names:
            # logging.info(f'Processing sheet: {sheet_name}')

            try:
                # Read excel file, skip first three rows
                df = pd.read_excel(os.path.join(base_path, excel_filename),
                                   sheet_name=sheet_name,
                                   skiprows=3,
                                   na_filter=False)

                # Keep rows where 'MP' column is not empty
                df = df[df['MP'].notna()]
                all_data.append(df)
                # logging.info(f'Successfully processed {sheet_name} with {len(df)} row')
            except Exception as e:
                # logging.error(f'Error processing sheet {sheet_name}: {str(e)}')
                continue
        if all_data:
            combine_df = pd.concat(all_data, ignore_index=True)
            combine_df['ATA/ATD'] = pd.to_datetime(combine_df['ATA/ATD'], dayfirst=True)
            combine_df.to_csv(os.path.join(base_path, csv_filename), index=False)

            return combine_df

        else:
            logging.warning('No data to combine')

    except Exception as e:
        logging.error(f'Error processing Excel file: {str(e)}')
        raise

    return pd.DataFrame()


def match_flight_bp_noise(bp_data: pd.DataFrame, combined_noise_data: pd.DataFrame, base_path) -> pd.DataFrame:
    """
    Match flights between best-practice Data and noise data based on aircraft type, measurement points and timestamps
    """
    matched_flight = []
    matched_bp_results = []
    matched_icao24 = []
    matched_callsign = []
    # Iterate best practice data by row
    for _, bp_row in bp_data.iterrows():
        mp = bp_row['MP']

        # Get corresponding noise data sheet
        mp_noise_data = combined_noise_data[combined_noise_data['MP'] == mp]

        if mp_noise_data.empty:
            logging.warning(f'No noise data found for measurement point {mp}')
            continue

        # Convert timestamps to datetime objects
        bp_time = pd.to_datetime(bp_row['ATA/ATD'])  # Adjust column name as needed
        noise_times = pd.to_datetime(mp_noise_data['ATA/ATD'], dayfirst=True)  # Time data e.g. "13.01.2022 01:55:39"

        # Match based on aircraft type
        noise_matches = mp_noise_data[
            (mp_noise_data['Flugzeugtyp'] == bp_row['Flugzeugtyp']) &
            (noise_times.dt.floor('min') == bp_time.floor('min'))
            ]

        # print(f'noise match: {noise_matches}')

        if not noise_matches.empty:
            matched_flight.append(noise_matches.iloc[0])
            matched_bp_results.append(bp_row['result_LAE_bestpractice'])
            matched_icao24.append(bp_row['icao24'])
            matched_callsign.append(bp_row['Callsign'])
            # logging.info(f"Found match at {mp} for aircraft type {bp_row['Flugzeugtyp']}")

        else:
            logging.warning(
                f"No matching flights found at {mp} for aircraft type {bp_row['Flugzeugtyp']} at time {bp_time}")

    if matched_flight:
        matched_flight_df = pd.DataFrame(matched_flight)
        matched_flight_df = matched_flight_df.reset_index(drop=True)
        matched_flight_df['best_practice_result_LAE'] = matched_bp_results
        matched_flight_df['icao24'] = matched_icao24
        matched_flight_df['callsign'] = matched_callsign
        # Convert German number format (comma as decimal separator) to standard format
        if 'T10 [s]' in matched_flight_df.columns:
            matched_flight_df['T10 [s]'] = matched_flight_df['T10 [s]'].apply(
                lambda x: float(str(x).replace(',', '.')) if pd.notnull(x) else None
            )

        return matched_flight_df

    return pd.DataFrame()


def match_weather_data(flight_data: pd.DataFrame, weather_data: pd.DataFrame):
    """
    Match flight data with corresponding weather data using linear interpolation

    Args:
        # base_path (str): Path of Data Folder
        flight_data (pd.DataFrame): DataFrame containing flight information
        weather_data (pd.DataFrame): DataFrame containing weather information

    Returns:
        pd.DataFrame: Original flight data with matched weather data
    """
    # Ensure timestamps are in datetime format
    flight_data['ATA/ATD'] = pd.to_datetime(flight_data['ATA/ATD'])
    weather_data['MESS_DATUM'] = pd.to_datetime(weather_data['MESS_DATUM'])

    # Create a copy of flight data to preserve all flights
    result_df = flight_data.copy()

    # Weather data columns to interpolate
    weather_cols = ['FF_10', 'DD_10', 'PP_10', 'TT_10', 'TM5_10', 'RF_10', 'TD_10']

    # Initialize weather columns with NaN values
    for col in weather_cols:
        result_df[col] = np.nan

    for idx, flight in flight_data.iterrows():
        try:
            flight_time = flight['ATA/ATD']
            # print(f'flight time: {flight_time}')
            # print(flight['Flugzeugtyp'])

            # Special debugging for E170 flights
            is_e170 = 'Flugzeugtyp' in flight and flight['Flugzeugtyp'] == 'E170'
            # if is_e170:
            #     print(f"Processing E170 flight at time: {flight_time}")

            # Define threshold for time difference (e.g., 30 seconds)
            time_threshold = pd.Timedelta(seconds=30)

            # Check for time match within a small threshold (e.g. 30 seconds)
            time_diffs = abs(weather_data['MESS_DATUM'] - flight_time).dt.total_seconds()
            # print(f'time difference: {min(time_diffs)}')

            # Find closest weather record
            closest_idx = time_diffs.idxmin()
            # print(f'closest index: {closest_idx}')
            min_time_diff = pd.Timedelta(seconds=abs(time_diffs[closest_idx]))
            # print(f'min time difference: {min_time_diff}')

            if min_time_diff <= time_threshold:
                # print(f'True')
                # If close match found, use those weather data values directly
                matched_weather = weather_data.loc[closest_idx]
                # print(f'matched weather: {matched_weather}')
                for col in weather_cols:
                    result_df.loc[idx, col] = matched_weather[col]

                # logging.info('Using exact match')
            else:
                # Find the closest weather measurements before and after the flight time
                before_weather = weather_data[weather_data['MESS_DATUM'] <= flight_time].iloc[-1]
                after_weather = weather_data[weather_data['MESS_DATUM'] > flight_time].iloc[0]

                # Calculate time difference in minutes
                time_diff_before = (flight_time - before_weather['MESS_DATUM']).total_seconds() / 60
                time_diff_after = (after_weather['MESS_DATUM'] - flight_time).total_seconds() / 60
                total_time_diff = time_diff_before + time_diff_after  # Here: 10 minutes

                # Calculate weights for interpolation
                weight_before = 1 - (time_diff_before / total_time_diff)
                weight_after = 1 - (time_diff_after / total_time_diff)

                # Interpolate weather values
                weather_values = {}

                for col in weather_cols:
                    interpolated_value = (
                            before_weather[col] * weight_before +
                            after_weather[col] * weight_after
                    )
                    result_df.loc[idx, col] = interpolated_value
                # logging.info('Using interpolated values')

        except Exception as e:
            logging.error(f'Error processing flight at index {idx}: {str(e)}')
            continue

        # Remove unnecessary columns
        columns_to_drop = ['Runway', 'SID/STAR', 'Triebwerk', 'AzB-Klasse']
        result_df = result_df.drop(columns=columns_to_drop, errors='ignore')

    return result_df


def prepare_merged_dataset(base_path: str):
    w_file_1 = os.path.join(base_path, 'produkt_zehn_min_ff_20200101_20221231_02014.csv')
    w_file_2 = os.path.join(base_path, 'produkt_zehn_min_tu_20200101_20221231_02014.csv')
    bp_file = os.path.join(base_path, 'best_practice_results.csv')

    # Step 1: Normalize timestamps
    best_practice_data, weather_data = normalize_timestamps(w_file_1, w_file_2, bp_file)

    # Step 2: Combine noise data sheets
    combined_noise_data = combine_noise_data_sheets(
        base_path,
        'Kopie von Fluglärmdaten 2022 (002).xlsx',
        'Kopie von Fluglärmdaten 2022 (002).csv'
    )

    # Step 3: Match flight data across weather, best-practice and noise data
    matched_flight_bp_noise = match_flight_bp_noise(best_practice_data, combined_noise_data, base_path)
    matched_flight_bp_noise_weather = match_weather_data(matched_flight_bp_noise, weather_data)
    matched_flight_filepath = os.path.join(base_path, 'Flights_Noise_BP_Weather.csv')
    matched_flight_bp_noise_weather.to_csv(matched_flight_filepath, index=False)

    return matched_flight_bp_noise_weather, matched_flight_filepath

