"""Interactive result browser for generated hostrada4py weather files.

This version is intentionally independent of exporter internals and can load
IDA ICE PRN, EnergyPlus EPW, Polysun CSV, SimStadt TMY3 and Modelica/
BuildingSystems table files.  If no file is explicitly supplied, it discovers
supported weather files in the current working directory automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display


_SUPPORTED_SUFFIXES = {".epw", ".prn", ".csv", ".tmy3", ".mos"}
_KNOWN_FILENAMES = (
    "HOSTRADA_IDA_ICE.prn",
    "HOSTRADA_Polysun.csv",
    "HOSTRADA_EnergyPlus.epw",
    "HOSTRADA_SimStadt.tmy3",
    "HOSTRADA_BuildingSystems.csv",
    "HOSTRADA_BuildingSystems.mos",
)


def _discover_weather_files(directory=".") -> list[Path]:
    """Return supported weather files, newest first."""
    directory = Path(directory)
    found: dict[Path, Path] = {}

    for name in _KNOWN_FILENAMES:
        path = directory / name
        if path.is_file():
            found[path.resolve()] = path

    try:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES:
                # Do not offer the IDA location metadata file; it is not a
                # tabular weather-data file.
                found[path.resolve()] = path
    except OSError:
        pass

    return sorted(
        found.values(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )


def _looks_like_modelica_table(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first = handle.readline().strip()
            second = handle.readline().strip()
        return first == "#1" and second.lower().startswith("double ")
    except OSError:
        return False


def _read_modelica_table(path: Path) -> pd.DataFrame:
    """Read a Modelica table written as '#1' + 'double tab1(...)' + CSV rows."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().strip()
        declaration = handle.readline().strip()

    if first != "#1":
        raise ValueError("Not a Modelica table file (missing '#1' header).")

    match = re.search(r"\((\d+)\s*,\s*(\d+)\)", declaration)
    declared_columns = int(match.group(2)) if match else None

    frame = pd.read_csv(path, skiprows=2, header=None)
    if frame.empty:
        return frame

    ncols = len(frame.columns)
    if declared_columns is not None and declared_columns != ncols:
        # Do not fail on a harmless declaration mismatch; keep the parsed data.
        pass

    names = ["seconds"] + [f"value_{i}" for i in range(1, ncols)]
    if ncols > 1:
        parsed_time = pd.to_datetime(frame.iloc[:, 1], errors="coerce")
        if parsed_time.notna().mean() >= 0.5:
            names[1] = "time"
    frame.columns = names
    return frame


def _read_ida_prn(path: Path) -> pd.DataFrame:
    """Read both the legacy headerless IDA PRN and newer tabular PRN output."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first_nonempty = ""
        for line in handle:
            if line.strip():
                first_nonempty = line.strip()
                break

    if not first_nonempty:
        return pd.DataFrame()

    # Newer hostrada4py export: named columns, normally tab-separated.
    if any(ch.isalpha() for ch in first_nonempty):
        for separator in ("\t", ",", ";", r"\s+"):
            try:
                frame = pd.read_csv(path, sep=separator, engine="python")
                if len(frame.columns) > 1:
                    return frame
            except Exception:
                continue

    # Legacy IDA ICE format: seven whitespace-separated numeric columns,
    # without a header.  The first column is the hour index.
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if len(frame.columns) == 7:
        frame.columns = [
            "hour_index",
            "tas",
            "hurs",
            "sfcWind_direction",
            "sfcWind",
            "dni",
            "dhi",
        ]
    else:
        frame.columns = [f"value_{i}" for i in range(len(frame.columns))]
    return frame


def read_weather_table(path):
    """Read a generated weather file into a pandas DataFrame."""
    if path is None:
        raise ValueError(
            "No weather file was selected. Enter a filename or generate a weather file first."
        )

    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Weather-file path is not a file: {path}")

    suffix = path.suffix.lower()

    if suffix == ".epw":
        names = [
            "year", "month", "day", "hour", "minute", "flags", "tas", "tdew", "hurs", "ps",
            "extraterrestrial_horizontal", "extraterrestrial_direct", "horizontal_ir", "rsds", "dni", "dhi",
            "global_illuminance", "direct_illuminance", "diffuse_illuminance", "zenith_luminance",
            "sfcWind_direction", "sfcWind", "total_sky_cover", "opaque_sky_cover", "visibility",
            "ceiling_height", "present_weather_observation", "present_weather_codes", "precipitable_water",
            "aerosol_optical_depth", "snow_depth", "days_since_last_snow", "albedo", "liquid_precipitation_depth",
            "liquid_precipitation_quantity",
        ]
        frame = pd.read_csv(path, skiprows=8, header=None, names=names)
        frame["time"] = pd.to_datetime(
            dict(
                year=frame.year,
                month=frame.month,
                day=frame.day,
                hour=(pd.to_numeric(frame.hour, errors="coerce") - 1).clip(lower=0),
                minute=0,
            ),
            errors="coerce",
        )
        return frame

    if suffix == ".prn":
        return _read_ida_prn(path)

    if suffix == ".mos" or _looks_like_modelica_table(path):
        return _read_modelica_table(path)

    # Polysun, SimStadt and ordinary CSV/TSV files.
    for separator in (",", ";", "\t", r"\s+"):
        try:
            frame = pd.read_csv(path, sep=separator, comment="#", engine="python")
            if len(frame.columns) > 1:
                return frame
        except Exception:
            continue

    return pd.read_csv(path)


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in frame.columns}


def _time_index(frame: pd.DataFrame) -> pd.DatetimeIndex | None:
    lookup = _column_lookup(frame)

    # Two-column date/time formats used by IDA ICE and SimStadt.
    # This must be checked before a standalone "Time" column, otherwise
    # pandas would attach today's date to values such as "13:00".
    pairs = (
        ("date", "time", False),
        ("date (mm/dd/yyyy)", "time (hh:mm)", False),
    )
    for date_key, time_key, dayfirst in pairs:
        date_col = lookup.get(date_key)
        time_col = lookup.get(time_key)
        if date_col is not None and time_col is not None:
            combined = (
                frame[date_col].astype(str).str.strip()
                + " "
                + frame[time_col].astype(str).str.strip()
            )
            index = pd.DatetimeIndex(
                pd.to_datetime(combined, errors="coerce", dayfirst=dayfirst)
            )
            if index.notna().any():
                return index

    # Polysun commonly writes DateTime as DD.MM.YYYY HH:MM.
    datetime_col = lookup.get("datetime")
    if datetime_col is not None:
        values = frame[datetime_col].astype(str).str.strip()
        dot_date = values.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}")
        index = pd.DatetimeIndex(
            pd.to_datetime(
                values,
                errors="coerce",
                dayfirst=bool(dot_date.any()),
            )
        )
        if index.notna().any():
            return index

    # Other complete timestamp columns, case-insensitive.
    for candidate in ("timestamp", "date/time", "date_time", "time"):
        actual = lookup.get(candidate)
        if actual is None:
            continue
        values = frame[actual]
        # A column containing only clock times is not a complete time index.
        if candidate == "time":
            sample = values.astype(str).str.strip()
            if sample.str.match(r"^\d{1,2}:\d{2}(?::\d{2})?$").all():
                continue
        index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
        if index.notna().any():
            return index

    # EnergyPlus style year/month/day/hour columns.
    required = {"year", "month", "day", "hour"}
    if required <= set(lookup):
        year = pd.to_numeric(frame[lookup["year"]], errors="coerce")
        month = pd.to_numeric(frame[lookup["month"]], errors="coerce")
        day = pd.to_numeric(frame[lookup["day"]], errors="coerce")
        hour = (
            pd.to_numeric(frame[lookup["hour"]], errors="coerce")
            .fillna(1)
            .astype(int)
            .sub(1)
            .clip(lower=0)
        )
        return pd.DatetimeIndex(
            pd.to_datetime(
                dict(year=year, month=month, day=day, hour=hour),
                errors="coerce",
            )
        )

    # Legacy IDA PRN: relative hour number. Use a neutral reference year so
    # daily/monthly plots still work. This is a relative, not historical, date.
    hour_col = lookup.get("hour_index")
    if hour_col is not None:
        hours = pd.to_numeric(frame[hour_col], errors="coerce")
        if hours.notna().any():
            return pd.DatetimeIndex(
                pd.Timestamp("2001-01-01") + pd.to_timedelta(hours, unit="h")
            )

    return None


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return list(frame.select_dtypes(include=[np.number]).columns)


def visualize_weather_file(weather_file, columns=None, max_rows=8760):
    frame = weather_file if isinstance(weather_file, pd.DataFrame) else read_weather_table(weather_file)
    numeric = frame.select_dtypes("number")
    if columns:
        numeric = numeric[[column for column in columns if column in numeric]]
    index = _time_index(frame)
    if index is not None:
        numeric = numeric.set_axis(index)
    return numeric.iloc[:max_rows].plot(figsize=(12, 5), grid=True)


class WeatherFileVisualization:
    def __init__(self, weather_file=None, *, namespace=None, show=True):
        self.namespace = namespace
        self.weather_file = weather_file
        self.frame: pd.DataFrame | None = None
        self._build_widgets()
        self.load_button.on_click(self.load)
        self.variable.observe(self.draw, names="value")
        self.view.observe(self.draw, names="value")
        self.resample.observe(self.draw, names="value")
        if weather_file is not None:
            self.load()
        if show:
            self.show()

    def _build_widgets(self):
        initial_path = "" if isinstance(self.weather_file, pd.DataFrame) or self.weather_file is None else str(self.weather_file)
        self.path = widgets.Text(value=initial_path, description="Weather file:", layout=widgets.Layout(width="720px"))
        self.load_button = widgets.Button(description="Load / refresh", icon="refresh", button_style="info")
        self.variable = widgets.SelectMultiple(description="Variables:", layout=widgets.Layout(width="430px", height="150px"))
        self.view = widgets.ToggleButtons(
            options=[
                ("Time series", "series"),
                ("Daily profile", "profile"),
                ("Monthly means", "monthly"),
                ("Heatmap", "heatmap"),
                ("Distribution", "hist"),
                ("Table", "table"),
            ],
            value="series",
            description="View:",
        )
        self.resample = widgets.Dropdown(
            options=[("Hourly", None), ("Daily mean", "D"), ("Weekly mean", "W"), ("Monthly mean", "MS")],
            value=None,
            description="Aggregation:",
        )
        self.status = widgets.HTML()
        self.output = widgets.Output(layout=widgets.Layout(min_height="420px"))
        self.widget = widgets.VBox([
            widgets.HTML("<h3>Weather-file result explorer</h3>"),
            widgets.HBox([self.path, self.load_button]),
            self.variable,
            widgets.HBox([self.view, self.resample]),
            self.status,
            self.output,
        ])

    def show(self):
        display(self.widget)
        return self

    def load(self, _button=None):
        try:
            source = self.weather_file
            if self.path.value.strip():
                source = self.path.value.strip()

            if source is None or (isinstance(source, str) and not source.strip()):
                candidates = _discover_weather_files()
                if candidates:
                    source = candidates[0]
                    self.weather_file = source
                    self.path.value = str(source)
                else:
                    raise ValueError(
                        "No weather file selected and no supported weather file was found in the current directory."
                    )

            if isinstance(source, pd.DataFrame):
                frame = source.copy()
                source_label = "DataFrame"
            else:
                frame = read_weather_table(source)
                source_label = str(Path(source))
                self.weather_file = Path(source)
                self.path.value = str(source)

            self.frame = frame
            columns = numeric_columns(frame)
            preferred = [
                column
                for column in ("tas", "tdew", "hurs", "rsds", "dni", "dhi", "sfcWind")
                if column in columns
            ]
            self.variable.options = columns
            self.variable.value = tuple(preferred[:4] or columns[: min(4, len(columns))])
            self.status.value = (
                f"<b>{len(frame):,}</b> records and <b>{len(columns)}</b> numeric columns loaded "
                f"from <code>{source_label}</code>."
            )
            if self.namespace is not None:
                self.namespace.update(
                    weather_visualization=self,
                    weather_dataframe=frame,
                    weather_file=self.weather_file,
                )
            self.draw()
        except Exception as exc:
            self.status.value = f"<span style='color:#b00020'>Could not load weather data: {exc}</span>"

    def _selected_frame(self):
        if self.frame is None:
            return None
        columns = [column for column in self.variable.value if column in self.frame]
        if not columns:
            return None
        values = self.frame[columns].apply(pd.to_numeric, errors="coerce")
        index = _time_index(self.frame)
        if index is not None:
            values.index = index
            values = values[~values.index.isna()]
            if self.resample.value:
                values = values.resample(self.resample.value).mean()
        return values

    def draw(self, _change=None):
        values = self._selected_frame()
        if values is None or values.empty:
            return
        with self.output:
            clear_output(wait=True)
            import matplotlib.pyplot as plt

            view = self.view.value
            if view == "series":
                ax = values.plot(figsize=(13, 5), linewidth=1)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Time")
                plt.show()
            elif view == "profile":
                if not isinstance(values.index, pd.DatetimeIndex):
                    print("A time column is required for this view.")
                    return
                profile = values.groupby(values.index.hour).mean()
                ax = profile.plot(figsize=(11, 5), marker="o")
                ax.set_xlabel("Hour of day")
                ax.set_xticks(range(24))
                ax.grid(True, alpha=0.3)
                plt.show()
            elif view == "monthly":
                if not isinstance(values.index, pd.DatetimeIndex):
                    print("A time column is required for this view.")
                    return
                monthly = values.groupby(values.index.month).mean()
                ax = monthly.plot(kind="bar", figsize=(12, 5))
                ax.set_xlabel("Month")
                ax.grid(True, axis="y", alpha=0.3)
                plt.show()
            elif view == "heatmap":
                if not isinstance(values.index, pd.DatetimeIndex):
                    print("A time column is required for this view.")
                    return
                column = values.columns[0]
                pivot = pd.DataFrame({
                    "date": values.index.date,
                    "hour": values.index.hour,
                    "value": values[column].to_numpy(),
                }).pivot_table(index="date", columns="hour", values="value", aggfunc="mean")
                fig, ax = plt.subplots(figsize=(13, 6))
                image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
                ax.set_xlabel("Hour")
                ax.set_ylabel("Date index")
                ax.set_title(column)
                fig.colorbar(image, ax=ax)
                plt.show()
            elif view == "hist":
                axes = values.plot.hist(bins=40, alpha=0.55, figsize=(11, 5))
                axes.grid(True, alpha=0.3)
                plt.show()
            else:
                display(values.head(250))
                display(values.describe().T)


def setup_weather_file_visualization(namespace=None, weather_file=None, **kwargs):
    """Create the weather-file browser and resolve a sensible input source.

    Resolution order:
      1. explicit ``weather_file=...`` argument
      2. ``weather_file``, ``output_file`` or ``weather_dataframe`` in namespace
      3. newest supported weather file in the current working directory
    """
    if weather_file is None and namespace is not None:
        for key in ("weather_file", "output_file", "weather_dataframe"):
            candidate = namespace.get(key)
            if candidate is not None:
                weather_file = candidate
                break

    if weather_file is None:
        candidates = _discover_weather_files()
        if candidates:
            weather_file = candidates[0]

    return WeatherFileVisualization(weather_file, namespace=namespace, **kwargs)
