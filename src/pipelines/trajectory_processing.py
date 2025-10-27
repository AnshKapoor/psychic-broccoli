import logging
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import re
from datetime import datetime, timedelta
from typing import Tuple
from traffic.data import aircraft
from Trajectory_Preprocessing import preprocess_trajectory

from Data_Preparation import prepare_merged_dataset

'''Run Data_Preparation.py first'''
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
project_path = os.path.abspath('..')


def parse_dms(dms):
    """
    Converts a geographic coordinate in Degrees, Minutes and Seconds (DMS) format to Decimal Degrees (DD) format.

    Args:
        dms(str): DMS coordinate

    Returns:
        float: The decimal degree representing of the input DMS coordinate. Negative for South (S) and West (W),
               positive otherwise.

    Raises:
        ValueError: If the input string does not match the expected DMS format.
    """
    parts = re.match(r"(\d+)[°](\d+)'(\d+)?[\"']?([NSWE])", dms)
    if not parts:
        raise ValueError("Invalid DMS format!")
    degrees, minutes, seconds, direction = parts.groups()
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600

    return -dd if direction in 'SW' else dd


MP_COORDS = {
    'M 01': (parse_dms("9°45'37E"), parse_dms("52°27'13N")),
    'M 02': (parse_dms("9°45'02E"), parse_dms("52°27'57N")),
    'M 03': (parse_dms("9°47'25E"), parse_dms("52°27'60N")),
    'M 04': (parse_dms("9°48'48E"), parse_dms("52°27'19N")),
    'M 05': (parse_dms("9°49'57E"), parse_dms("52°28'05N")),
    'M 06': (parse_dms("9°37'31E"), parse_dms("52°27'14N")),
    'M 07': (parse_dms("9°34'55E"), parse_dms("52°27'40N")),
    'M 08': (parse_dms("9°32'48E"), parse_dms("52°28'07N")),
    'M 09': (parse_dms("9°37'09E"), parse_dms("52°28'06N")),
}

RUNWAY_COORDS = {
    '27R': {'start': (parse_dms("9°41'59E"), parse_dms("52°28'01N")),
            'end': (parse_dms("9°39'10E"), parse_dms("52°28'05N")),
            'color': 'red'},
    '27L': {'start': (parse_dms("9°42'40E"), parse_dms("52°27'15N")),
            'end': (parse_dms("9°40'36E"), parse_dms("52°27'18N")),
            'color': 'blue'}
}


def get_coordinate_range(mp: str, buffer: float = 0.01) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Obtain latitude and longitude ranges from measurement points

    Args:
        mp (str): Measurement point ID (e.g. 'M 01')
        buffer (float): Buffer size around coordinates

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float]]:
            ((max_lat, min_lat), (max_lon, min_lon))
    """
    if mp not in MP_COORDS:
        raise ValueError(f"Unknown measurement point: {mp}")

    # MP_COORDS format is (longitude, latitude)
    lon, lat = MP_COORDS[mp]

    # Create a coordinate range
    lat_range = (lat + buffer, lat - buffer)
    lon_range = (lon + buffer, lon - buffer)

    return lat_range, lon_range


def time_zone(date):
    """
    Check if the time is in summer or not.

    Args:
        date(datetime): The date to check.

    Returns:
        bool: True if the date is in summer time. False otherwise.
    """
    start_date = datetime(date.year, 3, 28)  # The start date is March 8th.
    end_date = datetime(date.year, 10, 30)  # The end date is October 30th.

    return start_date <= date <= end_date


def filter_entries_by_time(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """
    Filter entries by time only - returns the full trajectory within the time window

    Args:
        df (pd.DataFrame): Input DataFrame containing flight trajectory data.
        start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format
        end_time (str): End time in 'YYYY-MM-DD HH:MM:SS' format

    Returns:
         pd.DataFrame: Filtered DataFrame containing trajectories within specified time bounds
                       Returns empty DataFrame if no matches or an error
    """
    try:
        # Filter entries for specific locations and times
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        start_time = start_time.strftime('%d.%m.%Y %H:%M:%S')

        end_time = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        end_time = end_time.strftime('%d.%m.%Y %H:%M:%S')

        # Convert timestamp column to datetime if it's not already
        if len(df) > 0 and isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create temporal mask
        time_mask = ((df['timestamp'] >= start_time) &
                     (df['timestamp'] <= end_time))

        return df[time_mask]

    except Exception as e:
        logging.error(f"Error filtering entries by time: {e}")
        return pd.DataFrame()


def filter_entries_for_identification(df: pd.DataFrame,
                                      start_time: str,
                                      end_time: str,
                                      lat1: float,
                                      lat2: float,
                                      lon1: float,
                                      lon2: float) -> pd.DataFrame:
    """
    Filter entries for specific locations and times

    Args:
        df (pd.DataFrame): Input DataFrame containing flight trajectory data
        start_time (str): Start time in 'YYYY-MM-DD HH:MM:SS' format
        end_time (str): End time in 'YYYY-MM-DD HH:MM:SS' format
        lat1 (float): Upper latitude boundary
        lat2 (float): Lower latitude boundary
        lon1 (float): Upper longitude boundary
        lon2 (float): Lower longitude boundary

    Returns:
        pd.DataFrame: Filtered DataFrame containing trajectories within specified bounds
                      Returns empty DataFrame if no matches or on error
    """
    try:
        time_filtered_df = filter_entries_by_time(df, start_time, end_time)

        if time_filtered_df.empty:
            return pd.DataFrame()

        # Create mask for spatial bounds
        spatial_mask = ((df['latitude'] <= lat1) &
                        (df['latitude'] >= lat2) &
                        (df['longitude'] <= lon1) &
                        (df['longitude'] >= lon2))

        return time_filtered_df[spatial_mask]

    except Exception as e:
        logging.error(f"Error Filtering entries for identification: {e}")
        return pd.DataFrame()


def add_minutes_to_datetime(datetime_obj, minutes_to_add):
    """
    Add a specified number of minutes to the given datetime and gets the result as a formatted string

    Args:
        datetime_obj (datetime): A datetime object to which minutes will be added.
        minutes_to_add (int): Minutes to add.

    Returns:
        str: The update datetime in the format 'YYYY-MM-DD HH:MM:SS'

    Raises:
        TypeError: If the input datetime_obj is not a datetime object.
    """
    if not isinstance(datetime_obj, datetime):
        raise TypeError("Input must be a datetime object")

    # Calculate the new datetime by adding specified minutes
    new_datetime = datetime_obj + timedelta(minutes=minutes_to_add)

    # Format the result as a datetime string
    new_datetime_string = new_datetime.strftime('%Y-%m-%d %H:%M:%S')

    return new_datetime_string


def get_time_offsets(summer_time: bool) -> Tuple[int, int, int, int]:
    """
    Returns time offsets based on whether it's summer time or not.

    Args:
        summer_time (bool): Summer time -> True, otherwise -> False

    Returns:
        tuple: A tuple containing four time offsets for filtering and plotting time windows
    """

    return (-121, -140, -119, -100) if summer_time else (-61, -80, -59, -40)


def calculate_time_windows(tlasmax: datetime, offsets: Tuple[int, int, int, int]) -> Tuple[str, str, str, str]:
    """
    Calculates time windows for filtering based on given offsets.

    Args:
        tlasmax (datetime): Parsed maximum timestamp of the search.
        offsets (tuple): Time offsets tuple.

    Returns:
        tuple: Tuple containing four datetime strings:
                - begin_time: Start time for filtering
                - plot_begin_time: Start time for plotting
                - end_time: End time for filtering
                - plot_end_time: End time for plotting
    """
    x1, x2, x3, x4 = offsets
    begin_time = add_minutes_to_datetime(tlasmax, x1)
    plot_begin_time = add_minutes_to_datetime(tlasmax, x2)
    end_time = add_minutes_to_datetime(tlasmax, x3)
    plot_end_time = add_minutes_to_datetime(tlasmax, x4)

    return begin_time, plot_begin_time, end_time, plot_end_time


def process_aircraft(flight_search, potential_flight: pd.DataFrame, type_code: str, mode: str) -> Tuple[str, bool]:
    """
    Process an individual flight and checks if it matches criteria.

    Args:
        flight_search : Traffic flight object for searching.
        potential_flight (pd.DataFrame): Flight data from ADS-B
        type_code (str): Aircraft typecode to match.
        mode (str): 'Landung' or 'Start'

    Returns:
        Tuple: (callsign, match_found)
    """
    callsign = potential_flight['callsign'].iloc[0]
    icao24 = potential_flight['icao24'].iloc[0]

    # Get aircraft type from traffic.data
    aircraft_data = aircraft[icao24]
    req_typecode = aircraft_data.data['typecode'].iloc[0] if aircraft_data else 0

    match = (
        flight_search.takeoff_from('EDDV') if mode == 'Start'
        else flight_search.landing_at('EDDV') if mode == 'Landung'
        else False
    )

    if match and (req_typecode == type_code or req_typecode == 0):
        return callsign, True  # match: True

    return callsign, False  # match: False


def handle_no_match_scenario(typecode: str, match: bool):
    """
     Handles scenarios where no exact match is found.

     Args:
         typecode (str): Aircraft type code.
         match (bool): Whether the flight time matches.
    """
    if match:
        logging.warning(f'Matching flight time, but wrong plane type: {typecode}.')
    else:
        logging.warning('Matching flight time, but wrong mode')


def plot_over_time(df: pd.DataFrame, start_time: str, end_time: str, type_code: str, icao24: str):
    """
    PLot flight trajectory over time with runways and measurement points.

    Args:
        df (pd.DataFrame): Flight data.
        start_time (str): Start time for filtering.
        end_time (str): End time for filtering.
        type_code (str): Aircraft type code for the aircraft
        icao24 (str): ICAO24 identifier of the aircraft
    """
    try:
        # Filter the DataFrame for the specified time period
        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        filtered_df = df[mask]

        if filtered_df.empty:
            logging.warning('No data to plot for the specified time range.')
            return

        plt.figure(figsize=(12, 6))

        # Plot flight trajectory
        plt.plot(filtered_df['longitude'], filtered_df['latitude'], marker='o', linestyle='-', label='Flight Path')

        # Plot runways
        for runway, coords in RUNWAY_COORDS.items():
            color = coords['color']
            plt.plot(
                [coords['start'][0], coords['end'][0]],
                [coords['start'][1], coords['end'][1]],
                label=runway,
                color=coords['color']
            )

            # Marking the start and end points of runways
            plt.text(
                coords['start'][0], coords['start'][1],
                f'{runway} Start',
                fontsize=9,
                ha='right',
                color=color
            )
            plt.text(
                coords['end'][0], coords['end'][1],
                f'{runway} End',
                fontsize=9,
                ha='right',
                color=color
            )
        # Plot measurement points
        for name, (lon, lat) in MP_COORDS.items():
            plt.scatter(lon, lat)
            plt.text(lon, lat, name, fontsize=9)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Flight Trajectory of {type_code} ({icao24}) with Runways and Measurement Points')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error plotting flight over time: {e}")


def find_trajectory(merged_data_path: str) -> pd.DataFrame:
    """
    Find trajectory data for matched flights by combining noise data with trajectory data.

    This function:
    1. Loads and matches flight data from different data.
    2. Processes each matched flight to find its trajectory.
    3. Validates matches using ICAO24 codes and flight modes.
    4. Visualize trajectories for confirmed matches.

    Args:
        merged_data_path (str): Path to the merged data

    Returns:
        Tuple[str, str, str]. ICAO24 code, callsign, and flight data for matched flight
    """
    matched_flight = pd.read_csv(merged_data_path)
    matched_flight = pd.DataFrame(matched_flight)

    # Load joblib data
    joblib_data = {}
    base_path = os.path.dirname(merged_data_path)
    # for month in range(1, 2):
    #     try:
    #         month_name = {
    #             1: 'january'  # , 2: 'february', 3: 'march',
    #             # 4: 'april', 5: 'may', 6: 'june',
    #             # 7: 'july', 8: 'august', 9: 'september',
    #             # 10: 'october', 11: 'november', 12: 'december'
    #         }
    #         filename = f'data_2022_{month_name[month]}.joblib'
    #         joblib_data[month] = joblib.load(os.path.join(base_path, filename))
    #     except FileNotFoundError:
    #         logging.warning(f'Can not find joblib file in {month}')
    #         continue

    all_processed_trajectories = []

    for idx, flight in matched_flight.iterrows():
        try:
            # Extract flight information from matched data
            mp = flight['MP']

            tlasmax = pd.to_datetime(flight['TLASmax'], dayfirst=True)
            mode = flight['A/D']
            type_code = flight['Flugzeugtyp']
            month = tlasmax.month

            # if month not in joblib_data:
            #     logging.warning(f'No trajectory data available for month: {month}')
            #     continue

            # else:
            try:
                month_name = {
                    1: 'january'  # , 2: 'february', 3: 'march',
                    # 4: 'april', 5: 'may', 6: 'june',
                    # 7: 'july', 8: 'august', 9: 'september',
                    # 10: 'october', 11: 'november', 12: 'december'
                }
                filename = f'data_2022_{month_name[month]}.joblib'
                joblib_data[month] = joblib.load(os.path.join(base_path, filename))
            except FileNotFoundError:
                logging.warning(f'Can not find joblib file in {month}')
                continue

            # Get coordinate ranges from the measurement point
            latitude_range, longitude_range = get_coordinate_range(mp, buffer=0.01)

            # Get trajectory data for specific month
            monthly_data = joblib_data[month]

            # Search window around tlamax
            summer_time = time_zone(tlasmax)
            time_offsets = get_time_offsets(summer_time)
            begin_time, plot_begin_time, end_time, plot_end_time = calculate_time_windows(tlasmax, time_offsets)

            # Filter trajectory data by time window
            filtered_data = monthly_data.between(begin_time, end_time)

            # Filter trajectory data by geographical boundaries
            nearby_trajectory = filter_entries_for_identification(
                filtered_data.data,
                begin_time,
                end_time,
                latitude_range[0],
                latitude_range[1],
                longitude_range[0],
                longitude_range[1]
            )
            nearby_trajectory = nearby_trajectory.reset_index(drop=True)

            #  print(f'nearby trajectory: {nearby_trajectory}')

            if not nearby_trajectory.empty:
                logging.info(f"Found trajectory for flight at {flight['TLASmax']}")

                # Process each unique aircraft in the filtered trajectory data
                unique_icao24 = nearby_trajectory['icao24'].unique()

                for index, code in enumerate(unique_icao24):
                    # Get flight data for specific ICAO24 code
                    potential_flight = nearby_trajectory[nearby_trajectory['icao24'] == code]
                    icao24 = potential_flight['icao24'].iloc[0]
                    flight = monthly_data[code]  # e.g.: Flight(icao24='4caa4f', callsign='ABR581')
                    flight_search = flight.between(begin_time, end_time)

                    # Verify aircraft type and flight mode
                    callsign, match = process_aircraft(flight_search, potential_flight, type_code, mode)

                    # Plot trajectory if match found
                    if match:
                        # Get the complete trajectory for this time window
                        complete_trajectory = filtered_data.data[filtered_data.data['icao24'] == code].reset_index(
                            drop=True)
                        # print(f'complete trajectory: {complete_trajectory}')

                        if not complete_trajectory.empty:
                            processed_trajectory = preprocess_trajectory(complete_trajectory, target_length=100)
                            # print(f'processed trajectory: {processed_trajectory}')
                            if not processed_trajectory.empty:
                                processed_trajectory['icao24'] = icao24
                                processed_trajectory['callsign'] = callsign
                                processed_trajectory['Flugzeugtyp'] = type_code
                                processed_trajectory['mode'] = mode

                                # plot_over_time(processed_trajectory, plot_begin_time, plot_end_time,
                                # type_code, icao24)
                                all_processed_trajectories.append(processed_trajectory)
                        else:
                            logging.warning(f"No complete trajectory data found for {callsign} ({icao24})")

                    # # Handle case when no match is found after checking all aircraft
                    # if index == len(unique_icao24) - 1:
                    #     handle_no_match_scenario(type_code, match)
            else:
                logging.warning(f"No trajectory found for flight at {flight['TLASmax']}")

        except Exception as e:
            logging.error(f"Error processing flight {idx}: {e}")
            continue

    if all_processed_trajectories:
        final_dataset = pd.concat(all_processed_trajectories, ignore_index=True)
        return final_dataset
    else:
        return pd.DataFrame()



