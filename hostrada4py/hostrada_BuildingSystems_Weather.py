"""Modelica BuildingSystems weather table export."""
from __future__ import annotations

from pathlib import Path

from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostradaPoint import extract_multiple_values_for_point


def ensure_target(output_file, default_name):
    """Return the output path and create its parent directory if needed."""
    path = Path(output_file or default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def weather_dataframe(
    lon,
    lat,
    start,
    end,
    variables,
    *,
    tz="UTC",
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


def create_buildingsystems_weather_file(
    lon,
    lat,
    start,
    end,
    output_file="HOSTRADA_BuildingSystems.mos",
    *,
    tz="UTC",
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
    out = df.copy()
    out.insert(0, "seconds", range(0, len(out) * 3600, 3600))
    path = ensure_target(output_file, "HOSTRADA_BuildingSystems.mos")
    with path.open("w", encoding="utf-8") as f:
        f.write("#1\n")
        f.write(f"double tab1({len(out)},{len(out.columns)})\n")
        out.to_csv(f, index=False, header=False)
    return path


create_weather_file = create_buildingsystems_weather_file

# Original notebook/API alias preserved.
create_buildingsystems_csv_weather_file = create_buildingsystems_weather_file
