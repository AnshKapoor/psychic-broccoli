# Noise Simulation Clarifications

1. Should we group aircraft by ICAO type (`B738`, `E190`, `E195`, `E75L`) or by NPD ID (`CF567B`, `CF3410E`, `CF348E`) for both ground truth and cluster evaluation?

2. If two ICAO types share one NPD table (for example `E190` and `E195` -> `CF3410E`), should we:
   a. combine them into one noise group, or
   b. keep them separate with the same NPD but different flight profiles and spectral classes? keep separate

3. When ICAO types share NPD but differ in profile mapping, which profile should we treat as authoritative:
   a. per-ICAO profile (`EMB190` vs `EMB195`), or This one
   b. one profile per NPD group?

4. For spectral classes, should we always apply mapping per ICAO + operation (`Landung`/`Start`) even when NPD is shared? Spectral classes differ for arrival and departure

5. For `Flight_subtracks.csv`, should we assume `Nr.day` is already weighted by flights per subtrack (current behavior), or should we set `Nr.day = 1` and apply scaling later in aggregation?

6. For ground truth coverage, should we include only mapped top-10 ICAO types in `NPD_TYPE_MAP`, or should we include all flights with fallback/reference where mapping is missing?

7. Should we keep `Nr.night = 0` with all flights treated as day operations, or should we derive day/night weights from timestamps?
Assume all flights were during day Should be treated as 0

8. For final decision-making, should we use only `MAE(cumulative_res)` over 9 receiver points, or both MAE and max absolute error?

9. Should we report results only overall, or both overall and per flow (`Start_09L`, `Landung_27R`, etc.)?

10. For unmatched aircraft types, should we:
    a. skip with warning (current behavior),
    b. fail fast, or
    c. apply fallback defaults automatically?
