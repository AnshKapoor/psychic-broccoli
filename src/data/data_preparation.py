"""Utility helpers for assembling the training dataset.

This module prepares three heterogeneous data sources (best-practice noise
measurements, on-site sensor recordings, and weather observations) so they can
be merged into a single table that downstream pipelines can consume. The helper
functions normalise timestamp formats, aggregate Excel worksheets, cross-match
records across sources, and finally export a fully combined CSV file.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd

# Configure logging to provide lightweight progress updates while preparing
# the dataset. The configuration is intentionally simple to avoid overriding
# user-specific logging settings when this module is imported.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
project_path = os.path.abspath("..")


def normalize_timestamps(
    weather_file_1: str,
    weather_file_2: str,
    bp_results_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standardise timestamp columns from weather and best-practice sources.

    Parameters
    ----------
    weather_file_1:
        CSV file containing temperature, pressure, and humidity measurements.
    weather_file_2:
        CSV file containing wind-related measurements recorded at the same
        station as ``weather_file_1``.
    bp_results_file:
        CSV file containing the best-practice noise measurements.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple consisting of the cleaned best-practice data and the weather
        observations filtered to the year 2022.
    """

    # Read the weather data from the two separate files provided by the DWD.
    df_temporal = pd.read_csv(weather_file_1, delimiter=";")
    df_wind = pd.read_csv(weather_file_2, delimiter=";")

    # Concatenate both DataFrames along the columns and drop duplicate columns
    # introduced by overlapping metadata.
    weather_data = pd.concat([df_temporal, df_wind], axis=1)
    weather_data = weather_data.loc[:, ~weather_data.columns.duplicated()]

    # The best-practice dataset contains the manually validated measurements
    # for which we later want to find flight trajectories.
    best_practice_result = pd.read_csv(bp_results_file, delimiter=";")

    # --- Normalise the best-practice timestamps ---------------------------------
    bp_data = best_practice_result.copy()
    bp_timestamp_col = "ATA/ATD"
    bp_data[bp_timestamp_col] = bp_data[bp_timestamp_col].astype(str)
    bp_data[bp_timestamp_col] = pd.to_datetime(
        bp_data[bp_timestamp_col],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Ordering by measurement point and timestamp simplifies downstream merges.
    bp_data = bp_data.sort_values(by=["MP", "ATA/ATD"])

    # --- Normalise the weather timestamps ---------------------------------------
    weather_copy = weather_data.copy()
    weather_timestamp_col = "MESS_DATUM"
    weather_copy[weather_timestamp_col] = pd.to_numeric(weather_copy[weather_timestamp_col], errors="coerce")
    weather_copy[weather_timestamp_col] = pd.to_datetime(
        weather_copy[weather_timestamp_col],
        format="%Y%m%d%H%M",
        errors="coerce",
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Keep only measurements recorded in 2022 because the best-practice data is
    # limited to that year.
    weather_copy[weather_timestamp_col] = pd.to_datetime(weather_copy[weather_timestamp_col], errors="coerce")
    w_data_2022 = weather_copy[weather_copy[weather_timestamp_col].dt.year == 2022].reset_index(drop=True)

    return bp_data, w_data_2022


def combine_noise_data_sheets(base_path: str, excel_filename: str, csv_filename: str) -> pd.DataFrame:
    """Merge multiple Excel worksheets with noise measurements into a CSV file.

    Parameters
    ----------
    base_path:
        Directory containing the raw Excel workbook.
    excel_filename:
        File name of the Excel workbook that stores noise measurements per
        measurement point in separate sheets.
    csv_filename:
        Name of the CSV file that should receive the merged contents.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame of all sheets. An empty DataFrame is returned
        if the Excel file contained no usable data.
    """

    # The workbook includes one sheet per measurement point. We accumulate all
    # individual DataFrames in a list to concatenate them at the end.
    all_data: list[pd.DataFrame] = []

    try:
        noise_data = os.path.join(base_path, excel_filename)
        noise_data_xls = pd.ExcelFile(noise_data)

        # Iterate over every sheet so we can extract the measurements from all
        # locations covered in the Excel workbook.
        for sheet_name in noise_data_xls.sheet_names:
            try:
                df = pd.read_excel(
                    os.path.join(base_path, excel_filename),
                    sheet_name=sheet_name,
                    skiprows=3,
                    na_filter=False,
                )

                # Ignore empty rows—those typically contain header information
                # or trailing whitespace in the original workbook.
                df = df[df["MP"].notna()]
                all_data.append(df)
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.error("Error processing sheet %s: %s", sheet_name, exc)
                continue

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df["ATA/ATD"] = pd.to_datetime(combined_df["ATA/ATD"], dayfirst=True)
            combined_df.to_csv(os.path.join(base_path, csv_filename), index=False)
            return combined_df

        logging.warning("No data to combine from %s", excel_filename)

    except Exception as exc:  # pragma: no cover - file operations
        logging.error("Error processing Excel file %s: %s", excel_filename, exc)
        raise

    return pd.DataFrame()


def match_flight_bp_noise(
    bp_data: pd.DataFrame,
    combined_noise_data: pd.DataFrame,
    base_path: str,
) -> pd.DataFrame:
    """Align best-practice flights with noise measurements.

    Parameters
    ----------
    bp_data:
        DataFrame containing the best-practice measurements with timestamps and
        aircraft identifiers.
    combined_noise_data:
        DataFrame produced by :func:`combine_noise_data_sheets` containing
        sensor readings per measurement point.
    base_path:
        Base directory where intermediate files are stored. The argument is kept
        for compatibility with earlier callers and enables logging to reference
        the data location.

    Returns
    -------
    pd.DataFrame
        Matched noise measurements augmented with best-practice attributes. An
        empty DataFrame indicates that no matches were discovered.
    """

    # Retain the argument for compatibility with earlier interfaces even though
    # the path is not directly needed here.
    _ = base_path

    matched_flights: list[pd.Series] = []
    matched_bp_results: list[float] = []
    matched_icao24: list[str] = []
    matched_callsign: list[str] = []

    # Iterate best-practice rows one by one so we can locate the corresponding
    # noise sensor measurements for each flight event.
    for _, bp_row in bp_data.iterrows():
        mp = bp_row["MP"]

        # Select all noise measurements that belong to the current measurement
        # point.
        mp_noise_data = combined_noise_data[combined_noise_data["MP"] == mp]

        if mp_noise_data.empty:
            logging.warning("No noise data found for measurement point %s", mp)
            continue

        # Convert timestamps to comparable datetime objects before matching.
        bp_time = pd.to_datetime(bp_row["ATA/ATD"])
        noise_times = pd.to_datetime(mp_noise_data["ATA/ATD"], dayfirst=True)

        # Match by aircraft type and align within the same minute to account
        # for small recording offsets between systems.
        noise_matches = mp_noise_data[
            (mp_noise_data["Flugzeugtyp"] == bp_row["Flugzeugtyp"])
            & (noise_times.dt.floor("min") == bp_time.floor("min"))
        ]

        if not noise_matches.empty:
            matched_flights.append(noise_matches.iloc[0])
            matched_bp_results.append(bp_row["result_LAE_bestpractice"])
            matched_icao24.append(bp_row["icao24"])
            matched_callsign.append(bp_row["Callsign"])
        else:
            logging.warning(
                "No matching flights found at %s for aircraft type %s at time %s",
                mp,
                bp_row["Flugzeugtyp"],
                bp_time,
            )

    if matched_flights:
        matched_flight_df = pd.DataFrame(matched_flights).reset_index(drop=True)
        matched_flight_df["best_practice_result_LAE"] = matched_bp_results
        matched_flight_df["icao24"] = matched_icao24
        matched_flight_df["callsign"] = matched_callsign

        # Convert German decimal formatting (comma as decimal separator) to the
        # standard decimal point representation.
        if "T10 [s]" in matched_flight_df.columns:
            matched_flight_df["T10 [s]"] = matched_flight_df["T10 [s]"].apply(
                lambda value: float(str(value).replace(",", ".")) if pd.notnull(value) else None
            )

        return matched_flight_df

    return pd.DataFrame()


def match_weather_data(flight_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """Attach weather measurements to each flight via interpolation.

    Parameters
    ----------
    flight_data:
        DataFrame of matched flights which already includes best-practice and
        noise metadata.
    weather_data:
        Weather measurements with timestamps produced by
        :func:`normalize_timestamps`.

    Returns
    -------
    pd.DataFrame
        The input ``flight_data`` enriched with weather attributes for each
        flight timestamp.
    """

    # Ensure timestamps are comparable datetime objects.
    flight_data["ATA/ATD"] = pd.to_datetime(flight_data["ATA/ATD"])
    weather_data["MESS_DATUM"] = pd.to_datetime(weather_data["MESS_DATUM"])

    result_df = flight_data.copy()
    weather_cols = ["FF_10", "DD_10", "PP_10", "TT_10", "TM5_10", "RF_10", "TD_10"]

    # Prepare empty columns so later interpolation simply assigns numerical
    # values at the correct indices.
    for col in weather_cols:
        result_df[col] = np.nan

    time_threshold = pd.Timedelta(seconds=30)

    for idx, flight in flight_data.iterrows():
        try:
            flight_time = flight["ATA/ATD"]

            # Identify the closest weather observation and check whether it is
            # sufficiently close in time to use without interpolation.
            time_diffs = (weather_data["MESS_DATUM"] - flight_time).abs().dt.total_seconds()
            closest_idx = time_diffs.idxmin()
            min_time_diff = pd.Timedelta(seconds=abs(time_diffs[closest_idx]))

            if min_time_diff <= time_threshold:
                matched_weather = weather_data.loc[closest_idx]
                for col in weather_cols:
                    result_df.loc[idx, col] = matched_weather[col]
            else:
                # Otherwise linearly interpolate using the closest measurement
                # before and after the flight time.
                before_weather = weather_data[weather_data["MESS_DATUM"] <= flight_time].iloc[-1]
                after_weather = weather_data[weather_data["MESS_DATUM"] > flight_time].iloc[0]

                time_diff_before = (flight_time - before_weather["MESS_DATUM"]).total_seconds() / 60
                time_diff_after = (after_weather["MESS_DATUM"] - flight_time).total_seconds() / 60
                total_time_diff = time_diff_before + time_diff_after

                weight_before = 1 - (time_diff_before / total_time_diff)
                weight_after = 1 - (time_diff_after / total_time_diff)

                for col in weather_cols:
                    interpolated_value = (before_weather[col] * weight_before) + (after_weather[col] * weight_after)
                    result_df.loc[idx, col] = interpolated_value

        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error("Error processing flight at index %s: %s", idx, exc)
            continue

    # Drop metadata columns that are not required for modelling.
    columns_to_drop = ["Runway", "SID/STAR", "Triebwerk", "AzB-Klasse"]
    result_df = result_df.drop(columns=columns_to_drop, errors="ignore")

    return result_df


def prepare_merged_dataset(base_path: str) -> Tuple[pd.DataFrame, str]:
    """Create the fully merged dataset used throughout the project.

    Parameters
    ----------
    base_path:
        Directory containing all raw data files (weather, best-practice, and
        noise measurements).

    Returns
    -------
    Tuple[pd.DataFrame, str]
        The merged dataset as a DataFrame along with the path to the exported
        CSV file for external tooling.
    """

    w_file_1 = os.path.join(base_path, "produkt_zehn_min_ff_20200101_20221231_02014.csv")
    w_file_2 = os.path.join(base_path, "produkt_zehn_min_tu_20200101_20221231_02014.csv")
    bp_file = os.path.join(base_path, "best_practice_results.csv")

    # Step 1: Normalise timestamps to ensure consistent datetime formats.
    best_practice_data, weather_data = normalize_timestamps(w_file_1, w_file_2, bp_file)

    # Step 2: Merge the workbook containing raw noise measurements.
    combined_noise_data = combine_noise_data_sheets(
        base_path,
        "Kopie von Fluglärmdaten 2022 (002).xlsx",
        "Kopie von Fluglärmdaten 2022 (002).csv",
    )

    # Step 3: Join noise, best-practice, and weather data into one table.
    matched_flight_bp_noise = match_flight_bp_noise(best_practice_data, combined_noise_data, base_path)
    matched_flight_bp_noise_weather = match_weather_data(matched_flight_bp_noise, weather_data)
    matched_flight_filepath = os.path.join(base_path, "Flights_Noise_BP_Weather.csv")
    matched_flight_bp_noise_weather.to_csv(matched_flight_filepath, index=False)

    return matched_flight_bp_noise_weather, matched_flight_filepath

