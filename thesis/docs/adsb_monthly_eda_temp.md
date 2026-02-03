# ADS-B Monthly Files - Sample EDA (Temporary)

## April 2022 (adsb/data_2022_april.joblib)
- Rows: 22,140,392
- Columns (21): timestamp, icao24, latitude, longitude, groundspeed, track, vertical_rate, callsign, onground, alert, spi, squawk, altitude, geoaltitude, last_position, hour, firstseen, origin, lastseen, destination, day
- Timestamp range (UTC): 2022-03-31 22:00:00+00:00 to 2022-04-30 21:59:59+00:00
- Unique icao24: 618
- Unique callsigns: 494

Sample rows (first 5):
```
timestamp,icao24,latitude,longitude,groundspeed,track,vertical_rate,callsign,onground,alert,spi,squawk
2022-03-31 22:00:00+00:00,4bcc56,52.13478326797485,11.33623838424683,379.0,299.94873046875,,THY2NX,False,False,False,3215
2022-03-31 22:00:01+00:00,4bcc56,52.13478326797485,11.33623838424683,379.0,299.9513855058617,-2176.0,THY2NX,False,False,False,3215
2022-03-31 22:00:02+00:00,4bcc56,52.13703632354736,11.32988691329956,379.0,300.08056640625,,THY2NX,False,False,False,3215
2022-03-31 22:00:03+00:00,4bcc56,52.13800191879272,11.32722616195679,377.0,299.893798828125,,THY2NX,False,False,False,3215
2022-03-31 22:00:04+00:00,4bcc56,52.1379942813162,11.32721819196429,377.0,299.8956097345322,-2176.0,THY2NX,False,False,False,3215
```

## December 2022 (adsb/data_2022_december.joblib)
- Rows: 3,017,094
- Columns (17): timestamp, icao24, latitude, longitude, groundspeed, track, vertical_rate, callsign, onground, alert, spi, squawk, altitude, geoaltitude, last_position, serials, hour
- Timestamp range (UTC): 2022-11-30 22:00:01+00:00 to 2022-12-31 21:34:48+00:00
- Unique icao24: 509
- Unique callsigns: 397

Sample rows (first 5):
```
timestamp,icao24,latitude,longitude,groundspeed,track,vertical_rate,callsign,onground,alert,spi,squawk
2022-11-30 22:00:01+00:00,3c674d,51.705710200940146,9.842758178710938,410.0,335.18580300946485,-2560.0,DLH5TA,False,False,False,1000
2022-11-30 22:00:01+00:00,3c674d,51.705710200940146,9.842758178710938,410.0,335.18580300946485,-2560.0,DLH5TA,False,False,False,1000
2022-11-30 22:00:01+00:00,ac7ad2,52.49369993048199,9.305576869419644,377.0,245.20361516832514,2816.0,FDX6302,False,False,False,1000
2022-11-30 22:00:01+00:00,ac7ad2,52.49369993048199,9.305576869419644,377.0,245.20361516832514,2816.0,FDX6302,False,False,False,1000
2022-11-30 22:00:02+00:00,3c674d,51.708968857587394,9.840316772460938,410.0,335.059039626556,-2496.0,DLH5TA,False,False,False,1000
```
