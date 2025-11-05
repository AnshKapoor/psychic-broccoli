import pandas as pd
import numpy as np
import pytz
from traffic.data import aircraft



def match_noise_to_adsb(
    df_noise: str,
    adsb_parquet: str,
    out_noise_csv: str = None,
    out_traj_parquet: str = None,
    tol_sec: int = 10,
    buffer_frac: float = 0.5,
    window_min: int = 3
) -> (pd.DataFrame, pd.DataFrame):
    """
    Match noise measurements (already loaded in df_noise) to ADS-B trajectories.
    Expects df_noise to have either
      - a tz‐aware column 'TLASmax_UT' or
      - a naive 'TLASmax' in Europe/Berlin time
    
    On output, stamps df_noise with 'icao24' and 'callsign' (or NA if no match),
    and returns the concatenated ±window_min-minute ADS-B slices.
    """

    # 1) Ensure we have a UTC timestamp to match against
    df_noise = pd.read_csv(df_noise, sep=";")
    print(df_noise.columns)

    if "TLASmax_UTC" in df_noise.columns:
        # assume it's already tz-aware UTC
        df_noise["t_ref"] = pd.to_datetime(df_noise["TLASmax_UTC"], utc=True)
    else:
        # fallback: convert naive Berlin time → UTC
        import pytz
        df_noise["TLASmax"] = pd.to_datetime(df_noise["TLASmax"], dayfirst=True)
        ber = pytz.timezone("Europe/Berlin")
        df_noise["t_ref"] = (
            df_noise["TLASmax"]
              .dt.tz_localize(ber, ambiguous="infer")
              .dt.tz_convert(pytz.UTC)
        )

    # prepare output columns
    df_noise["icao24"]   = pd.NA
    df_noise["callsign"] = pd.NA

    # 2) load ADS-B data (already UTC)
    df_ads = pd.read_parquet(adsb_parquet)
    df_ads["timestamp"] = pd.to_datetime(df_ads["timestamp"], utc=True)

    # 3) MP coords & bounding‐box helper
    def parse_dms(dms_str):
        s = dms_str.replace("°"," ").replace("'"," ").replace("\""," ").strip()
        deg, minu, sec, hemi = s.split()
        dd = float(deg) + float(minu)/60 + float(sec)/3600
        return -dd if hemi in ("S","W") else dd

    mp_coords = {
      "M 01": (parse_dms("52°27'13\"N"), parse_dms("9°45'37\"E")),
      "M 02": (parse_dms("52°27'57\"N"), parse_dms("9°45'02\"E")),
      "M 03": (parse_dms("52°27'60\"N"), parse_dms("9°47'25\"E")),
      "M 04": (parse_dms("52°27'19\"N"), parse_dms("9°48'48\"E")),
      "M 05": (parse_dms("52°28'05\"N"), parse_dms("9°49'57\"E")),
      "M 06": (parse_dms("52°27'14\"N"), parse_dms("9°37'31\"E")),
      "M 07": (parse_dms("52°27'40\"N"), parse_dms("9°34'55\"E")),
      "M 08": (parse_dms("52°28'07\"N"), parse_dms("9°32'48\"E")),
      "M 09": (parse_dms("52°28'06\"N"), parse_dms("9°37'09\"E")),
    }

    def get_bbox(lat0, lon0, dist_m, buf_frac):
        r = dist_m * (1 + buf_frac)
        deg_lat = r / 111195.0
        deg_lon = deg_lat / np.cos(np.deg2rad(lat0))
        return lat0 - deg_lat, lat0 + deg_lat, lon0 - deg_lon, lon0 + deg_lon

    # 4) Loop over noise rows, find ±tol_sec match, extract ±window_min trajectory
    tol = pd.Timedelta(seconds=tol_sec)
    win = pd.Timedelta(minutes=window_min)
    trajs = []

    for i, row in df_noise.iterrows():
        t0   = row["t_ref"]
        mp   = row["MP"]
        dist = row["Abstand [m]"]
        lat0, lon0 = mp_coords.get(mp, (None,None))
        if lat0 is None:
            continue

        # spatial + time mask to identify flight
        lat_min, lat_max, lon_min, lon_max = get_bbox(lat0, lon0, dist, buffer_frac)
        m1 = (
          (df_ads.timestamp >=  t0 - tol) &
          (df_ads.timestamp <=  t0 + tol) &
          (df_ads.latitude  >= lat_min) &
          (df_ads.latitude  <= lat_max) &
          (df_ads.longitude >= lon_min) &
          (df_ads.longitude <= lon_max)
        )
        hits = df_ads.loc[m1]
        if hits.empty:
            continue

        # stamp the noise row
        icao24, callsign = hits.iloc[0][["icao24","callsign"]]
        
        # hits is going to be a dataframe 
        # if we have more than one icao24 code then we should also look if we have the correct aircraft type 

        # ac = aircraft.get(icao24)
        # typecode: typecode = ac.typecode
        # then you can check if the typecode matches the typecode in the noise_df 

        
        df_noise.at[i, "icao24"]   = icao24
        df_noise.at[i, "callsign"] = callsign

        # extract the full ±window slice
        m2 = (
          (df_ads.icao24   == icao24) &
          (df_ads.callsign == callsign) &
          (df_ads.timestamp >= t0 - win) &
          (df_ads.timestamp <= t0 + win)
        )
        slice6 = df_ads.loc[m2].copy()
        slice6["MP"]      = mp
        slice6["t_ref"]   = t0
        slice6["icao24"]   = icao24
        slice6["callsign"] = callsign
        trajs.append(slice6)

    # 5) finalize
    df_traj = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()

    # 6) optional write-out
    if out_noise_csv:
        df_noise.to_csv(out_noise_csv, index=False)
    if out_traj_parquet and not df_traj.empty:
        df_traj.to_parquet(out_traj_parquet, index=False)

    return df_noise, df_traj

#df = pd.read_csv("R:/MS_2022_68602240_DFG-EXC-SE2A-SONAR/NVM/Data Hannover/Landing_NoiseEvents_HAJ2024.csv", sep=';')
#adsb_df = pd.read_parquet("R:/MS_2022_68602240_DFG-EXC-SE2A-SONAR/NVM/Data Hannover/ADSB data opensky/data_HAJ_merged.parquet")


df_noise_aug, df_mini = match_noise_to_adsb(
    "R:/MS_2022_68602240_DFG-EXC-SE2A-SONAR/NVM/Data Hannover/Landing_NoiseEvents_HAJ2024_B738.csv",
    "R:/MS_2022_68602240_DFG-EXC-SE2A-SONAR/NVM/Data Hannover/ADSB data opensky/data_HAJ_merged.parquet",
    "R:/MS_2022_68602240_DFG-EXC-SE2A-SONAR/NVM/Data Hannover/Landing_NoiseEvents_HAJ2024_B738_with_callsign.csv",
    "R:/MS_2022_68602240_DFG-EXC-SE2A-SONAR/NVM/Data Hannover/ADSB data opensky/B738_trajectories_HAJ.parquet"
)
