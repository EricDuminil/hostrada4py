"""SimStadt/TMY3-style weather export."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostradaPoint import extract_multiple_values_for_point


def ensure_target(output_file, default_name):
    """Return the output path and create its parent directory if needed."""
    path = Path(output_file or default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def local_time(df, tz):
    """Return the dataframe time column converted from UTC to *tz*."""
    if "time" not in df.columns:
        raise KeyError("Weather dataframe contains no 'time' column.")

    times = pd.DatetimeIndex(pd.to_datetime(df["time"]))
    if times.tz is None:
        times = times.tz_localize("UTC")
    return times.tz_convert(tz)


def weather_dataframe(
    lon,
    lat,
    start,
    end,
    variables,
    *,
    tz="Europe/Berlin",
    altitude=0,
    apply_weather_correction=False,
    cache_dir="hostrada_cache",
    cache_strategy=None,
    subset_margin_cells=None,
    provider=None,
    verbose=True,
):
    """Load point weather data and add diffuse/direct solar radiation."""
    df = extract_multiple_values_for_point(
        variables=variables,
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        cache_dir=cache_dir,
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
        provider=provider,
        verbose=verbose,
    )

    if df.empty:
        raise ValueError("No weather data was returned for the requested period.")

    model = HostradaDiffuse(
        latitude=lat,
        longitude=lon,
        altitude=0.0 if altitude is None else float(altitude),
        tz=tz,
    )
    return model.estimate(
        df,
        apply_weather_correction=apply_weather_correction,
    )


def create_simstadt_weather_file(
    lon,
    lat,
    start,
    end,
    output_file="HOSTRADA_SimStadt.tmy3",
    *,
    tz="Europe/Berlin",
    altitude=0,
    apply_weather_correction=False,
    cache_dir="hostrada_cache",
    cache_strategy=None,
    subset_margin_cells=None,
    provider=None,
    verbose=True,
    **kwargs,
):
    required = ["tas", "tdew", "hurs", "ps", "sfcWind", "sfcWind_direction", "rsds", "clt"]
    df = weather_dataframe(
        lon,
        lat,
        start,
        end,
        required,
        tz=tz,
        altitude=altitude,
        apply_weather_correction=apply_weather_correction,
        cache_dir=cache_dir,
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
        provider=provider,
        verbose=verbose,
    )
    times = local_time(df, tz)
    out = df.copy()
    out.insert(0, "Date (MM/DD/YYYY)", times.strftime("%m/%d/%Y"))
    out.insert(1, "Time (HH:MM)", times.strftime("%H:%M"))
    path = ensure_target(output_file, "HOSTRADA_SimStadt.tmy3")
    out.to_csv(path, index=False)
    return path


create_weather_file = create_simstadt_weather_file
