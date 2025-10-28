"""High-level pipeline helpers for matching flights with trajectory data.

The routines in this module orchestrate the search for ADS-B trajectories that
align with previously matched noise events. They identify candidate flights,
validate aircraft types, and optionally visualise the resulting trajectories.
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from traffic.data import aircraft

from src.data.trajectory_preprocessing import preprocess_trajectory

# Configure logging at module import so pipeline runs report their progress.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_dms(dms: str) -> float:
    """Convert degrees-minutes-seconds coordinate strings to decimal degrees.

    Parameters
    ----------
    dms:
        Coordinate string such as ``"9°45'37E"``.

    Returns
    -------
    float
        Decimal-degree representation of the coordinate.

    Raises
    ------
    ValueError
        If ``dms`` does not match the expected pattern.
    """

    parts = re.match(r"(\d+)[°](\d+)'(\d+)?[\"']?([NSWE])", dms)
    if not parts:
        raise ValueError("Invalid DMS format!")
    degrees, minutes, seconds, direction = parts.groups()
    decimal_degrees = float(degrees) + float(minutes) / 60 + float(seconds) / 3600

    return -decimal_degrees if direction in "SW" else decimal_degrees


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
    """Return latitude and longitude boundaries for a measurement point."""

    if mp not in MP_COORDS:
        raise ValueError(f"Unknown measurement point: {mp}")

    lon, lat = MP_COORDS[mp]
    lat_range = (lat + buffer, lat - buffer)
    lon_range = (lon + buffer, lon - buffer)

    return lat_range, lon_range


def time_zone(date: datetime) -> bool:
    """Return ``True`` when ``date`` lies inside the daylight-saving period."""

    start_date = datetime(date.year, 3, 28)
    end_date = datetime(date.year, 10, 30)
    return start_date <= date <= end_date


def filter_entries_by_time(df: pd.DataFrame, start_time: str, end_time: str) -> pd.DataFrame:
    """Filter trajectories to a specific time window.

    Parameters
    ----------
    df:
        Flight trajectory observations ordered by timestamp.
    start_time, end_time:
        Time window in ``"YYYY-MM-DD HH:MM:SS"`` format.
    """

    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        if len(df) > 0 and isinstance(df["timestamp"].iloc[0], str):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        time_mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
        return df[time_mask]
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Error filtering entries by time: %s", exc)
        return pd.DataFrame()


def filter_entries_for_identification(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
    lat1: float,
    lat2: float,
    lon1: float,
    lon2: float,
) -> pd.DataFrame:
    """Limit trajectories to a spatial bounding box within a time window.

    Parameters
    ----------
    df:
        Raw trajectory DataFrame returned by the ``traffic`` library.
    start_time, end_time:
        Bounds of the temporal window expressed as ISO-formatted strings.
    lat1, lat2, lon1, lon2:
        Spatial boundaries centred around the measurement point of interest.
    """

    try:
        time_filtered_df = filter_entries_by_time(df, start_time, end_time)
        if time_filtered_df.empty:
            return pd.DataFrame()

        spatial_mask = (
            (time_filtered_df["latitude"] <= lat1)
            & (time_filtered_df["latitude"] >= lat2)
            & (time_filtered_df["longitude"] <= lon1)
            & (time_filtered_df["longitude"] >= lon2)
        )

        return time_filtered_df[spatial_mask]
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Error filtering entries for identification: %s", exc)
        return pd.DataFrame()


def add_minutes_to_datetime(datetime_obj: datetime, minutes_to_add: int) -> str:
    """Return a formatted string representing ``datetime_obj`` plus ``minutes_to_add``."""

    if not isinstance(datetime_obj, datetime):
        raise TypeError("Input must be a datetime object")

    # Calculate the new datetime by adding specified minutes
    new_datetime = datetime_obj + timedelta(minutes=minutes_to_add)

    # Format the result as a datetime string
    new_datetime_string = new_datetime.strftime('%Y-%m-%d %H:%M:%S')

    return new_datetime_string


def get_time_offsets(summer_time: bool) -> Tuple[int, int, int, int]:
    """Return offsets that define search windows around a matched noise event."""

    return (-121, -140, -119, -100) if summer_time else (-61, -80, -59, -40)


def calculate_time_windows(tlasmax: datetime, offsets: Tuple[int, int, int, int]) -> Tuple[str, str, str, str]:
    """Translate offsets into formatted timestamps for searching and plotting."""

    begin_offset, plot_begin_offset, end_offset, plot_end_offset = offsets
    begin_time = add_minutes_to_datetime(tlasmax, begin_offset)
    plot_begin_time = add_minutes_to_datetime(tlasmax, plot_begin_offset)
    end_time = add_minutes_to_datetime(tlasmax, end_offset)
    plot_end_time = add_minutes_to_datetime(tlasmax, plot_end_offset)
    return begin_time, plot_begin_time, end_time, plot_end_time


def process_aircraft(
    flight_search: Any,
    potential_flight: pd.DataFrame,
    type_code: str,
    mode: str,
) -> Tuple[str, bool]:
    """Verify whether a candidate ADS-B flight matches the expected aircraft type."""

    callsign = potential_flight["callsign"].iloc[0]
    icao24 = potential_flight["icao24"].iloc[0]

    aircraft_data = aircraft[icao24]
    req_typecode = ""
    if aircraft_data is not None and hasattr(aircraft_data, "data"):
        data_frame = getattr(aircraft_data, "data")
        if isinstance(data_frame, pd.DataFrame) and "typecode" in data_frame.columns and not data_frame.empty:
            req_typecode = data_frame["typecode"].iloc[0]

    match = (
        flight_search.takeoff_from("EDDV") if mode == "Start"
        else flight_search.landing_at("EDDV") if mode == "Landung"
        else False
    )

    if match and (req_typecode == type_code or req_typecode == ""):
        return callsign, True

    return callsign, False


def handle_no_match_scenario(typecode: str, match: bool) -> None:
    """Log a helpful message when the aircraft type or mode does not match."""

    if match:
        logging.warning("Matching flight time, but wrong plane type: %s.", typecode)
    else:
        logging.warning("Matching flight time, but wrong mode")


def plot_over_time(df: pd.DataFrame, start_time: str, end_time: str, type_code: str, icao24: str) -> None:
    """Visualise a processed trajectory alongside runway and measurement points."""

    try:
        start_dt = pd.to_datetime(start_time)
        end_dt = pd.to_datetime(end_time)

        if len(df) > 0 and not isinstance(df["timestamp"].iloc[0], pd.Timestamp):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
        filtered_df = df[mask]

        if filtered_df.empty:
            logging.warning("No data to plot for the specified time range.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(filtered_df["longitude"], filtered_df["latitude"], marker="o", linestyle="-", label="Flight Path")

        for runway, coords in RUNWAY_COORDS.items():
            color = coords["color"]
            plt.plot(
                [coords["start"][0], coords["end"][0]],
                [coords["start"][1], coords["end"][1]],
                label=runway,
                color=color,
            )
            plt.text(coords["start"][0], coords["start"][1], f"{runway} Start", fontsize=9, ha="right", color=color)
            plt.text(coords["end"][0], coords["end"][1], f"{runway} End", fontsize=9, ha="right", color=color)

        for name, (lon, lat) in MP_COORDS.items():
            plt.scatter(lon, lat)
            plt.text(lon, lat, name, fontsize=9)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Flight Trajectory of {type_code} ({icao24}) with Runways and Measurement Points")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - visualisation helper
        logging.error("Error plotting flight over time: %s", exc)


def find_trajectory(merged_data_path: str) -> pd.DataFrame:
    """Identify ADS-B trajectories that correspond to merged noise events.

    Parameters
    ----------
    merged_data_path:
        Path to the CSV file produced by :func:`prepare_merged_dataset`.

    Returns
    -------
    pd.DataFrame
        Concatenated collection of processed trajectories that match the
        flights stored in the merged dataset. An empty DataFrame indicates that
        no matching trajectories could be located.
    """

    matched_flight = pd.read_csv(merged_data_path)
    matched_flight = pd.DataFrame(matched_flight)

    joblib_data: Dict[int, Any] = {}
    base_path = os.path.dirname(merged_data_path)
    all_processed_trajectories: List[pd.DataFrame] = []

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
                longitude_range[1],
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
                            if not processed_trajectory.empty:
                                processed_trajectory["icao24"] = icao24
                                processed_trajectory["callsign"] = callsign
                                processed_trajectory["Flugzeugtyp"] = type_code
                                processed_trajectory["mode"] = mode

                                # plot_over_time(processed_trajectory, plot_begin_time, plot_end_time,
                                # type_code, icao24)
                                all_processed_trajectories.append(processed_trajectory)
                        else:
                            logging.warning(f"No complete trajectory data found for {callsign} ({icao24})")

            else:
                logging.warning(f"No trajectory found for flight at {flight['TLASmax']}")

        except Exception as e:
            logging.error(f"Error processing flight {idx}: {e}")
            continue

    if all_processed_trajectories:
        final_dataset = pd.concat(all_processed_trajectories, ignore_index=True)
        return final_dataset

    return pd.DataFrame()



