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
import csv

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


def _read_tmy3(path: Path) -> pd.DataFrame:
    """Read a SimStadt/TMY3 file.

    A conventional TMY3 file starts with one metadata record (station id,
    station name, timezone, latitude, longitude, elevation, ...).  The actual
    weather-column header follows on the next line.  Some hostrada4py variants
    omit the metadata record, so this reader detects both layouts.
    """
    lines: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            lines.append((line_number, line.rstrip("\r\n")))
            if len(lines) >= 3:
                break

    if not lines:
        return pd.DataFrame()

    def _header_like(line: str) -> bool:
        lower = line.lower()
        return (
            "date" in lower
            and "time" in lower
            and any(token in lower for token in ("tas", "temperature", "dry", "hurs", "humidity", "wind", "rsds", "solar"))
        )

    # Standard TMY3: line 1 = station metadata, line 2 = weather header.
    # Simpler hostrada4py exports can start directly with the weather header.
    if len(lines) >= 2 and not _header_like(lines[0][1]) and _header_like(lines[1][1]):
        header_line_number, header_line = lines[1]
    else:
        header_line_number, header_line = lines[0]

    counts = {sep: header_line.count(sep) for sep in (",", ";", "\t")}
    separator = max(counts, key=counts.get)
    if counts[separator] == 0:
        separator = ","

    # Semicolon-separated European exports often use decimal commas; the
    # standard comma-separated TMY3 representation uses decimal points.
    decimal = "," if separator == ";" else "."

    frame = pd.read_csv(
        path,
        skiprows=header_line_number,
        sep=separator,
        decimal=decimal,
        engine="python",
        encoding="utf-8-sig",
    )

    # Defensive cleanup: malformed files can contain a repeated header row.
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    date_col = lookup.get("date (mm/dd/yyyy)") or lookup.get("date")
    time_col = lookup.get("time (hh:mm)") or lookup.get("time")
    if date_col is not None and time_col is not None and not frame.empty:
        repeated = (
            frame[date_col].astype(str).str.strip().str.lower().eq(str(date_col).strip().lower())
            | frame[time_col].astype(str).str.strip().str.lower().eq(str(time_col).strip().lower())
        )
        if repeated.any():
            frame = frame.loc[~repeated].reset_index(drop=True)

    return _promote_numeric_columns(frame)


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
        return _promote_numeric_columns(frame)

    if suffix == ".prn":
        return _promote_numeric_columns(_read_ida_prn(path))

    if suffix == ".tmy3":
        return _read_tmy3(path)

    if suffix == ".mos" or _looks_like_modelica_table(path):
        return _promote_numeric_columns(_read_modelica_table(path))

    # Polysun, SimStadt and ordinary CSV/TSV files.  Detect the delimiter
    # from the header instead of accepting the first separator that happens
    # to create multiple columns.  This matters especially for European
    # semicolon files with decimal commas: parsing those with sep="," can
    # create many bogus text columns and previously led to "0 numeric columns".
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample_lines = []
        for line in handle:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            sample_lines.append(line)
            if len(sample_lines) >= 8:
                break
    sample = "".join(sample_lines)

    separators = []
    if sample:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            separators.append(dialect.delimiter)
        except csv.Error:
            first = sample_lines[0] if sample_lines else ""
            counts = {sep: first.count(sep) for sep in (",", ";", "\t")}
            guessed = max(counts, key=counts.get)
            if counts[guessed] > 0:
                separators.append(guessed)

    for separator in (",", ";", "\t"):
        if separator not in separators:
            separators.append(separator)

    best_frame = None
    best_score = (-1, -1)
    for separator in separators:
        try:
            decimal = "," if separator == ";" else "."
            frame = pd.read_csv(
                path,
                sep=separator,
                decimal=decimal,
                comment="#",
                engine="python",
                encoding="utf-8-sig",
            )
            if len(frame.columns) <= 1:
                continue
            promoted = _promote_numeric_columns(frame)
            n_numeric = len(promoted.select_dtypes(include=[np.number]).columns)
            # Prefer parses with useful numeric weather columns, then more
            # columns.  This rejects the classic decimal-comma misparse.
            score = (n_numeric, len(promoted.columns))
            if score > best_score:
                best_frame = promoted
                best_score = score
        except Exception:
            continue

    if best_frame is not None:
        return best_frame

    return _promote_numeric_columns(pd.read_csv(path))


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column).strip().lower(): column for column in frame.columns}


_NON_NUMERIC_COLUMN_NAMES = {
    "date",
    "time",
    "datetime",
    "timestamp",
    "date/time",
    "date_time",
    "date (mm/dd/yyyy)",
    "time (hh:mm)",
    "flags",
}


def _promote_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert number-like text columns to numeric values.

    Some TMY3/SimStadt files are read by pandas with every column typed as
    ``object`` even though the weather columns contain numbers.  This helper
    promotes such columns while deliberately leaving date/time columns alone.
    Both decimal points and decimal commas are supported.
    """
    frame = frame.copy()

    for column in frame.columns:
        key = str(column).strip().lower()
        if key in _NON_NUMERIC_COLUMN_NAMES:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue

        values = frame[column].astype("string").str.strip()
        nonempty = values.notna() & ~values.str.lower().isin(
            {"", "nan", "none", "null", "na", "n/a", "-"}
        )
        if not nonempty.any():
            continue

        cleaned = values.str.replace("\u2212", "-", regex=False)
        # Decimal commas are common in European CSV-like exports.  Only
        # translate commas when this cannot be confused with a thousands
        # separator containing decimal points as well.
        comma_mask = cleaned.str.contains(",", regex=False, na=False)
        dot_mask = cleaned.str.contains(".", regex=False, na=False)
        if comma_mask.any() and not dot_mask.any():
            cleaned = cleaned.str.replace(",", ".", regex=False)

        parsed = pd.to_numeric(cleaned, errors="coerce")
        coverage = parsed[nonempty].notna().mean()
        if coverage >= 0.80:
            frame[column] = parsed

    return frame


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
            date_values = pd.to_datetime(
                frame[date_col].astype(str).str.strip(),
                errors="coerce",
                dayfirst=dayfirst,
            )
            time_values = frame[time_col].astype(str).str.strip()

            # TMY3 commonly uses 01:00 ... 24:00.  pandas cannot directly
            # parse 24:00 as a clock time, so construct the offset manually.
            parts = time_values.str.extract(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
            hours = pd.to_numeric(parts[0], errors="coerce")
            minutes = pd.to_numeric(parts[1], errors="coerce").fillna(0)
            seconds = pd.to_numeric(parts[2], errors="coerce").fillna(0)
            valid_clock = hours.between(0, 24, inclusive="both") & minutes.between(0, 59, inclusive="both")
            if valid_clock.mean() >= 0.8:
                day_add = hours.eq(24).astype(int)
                normalized_hours = hours.where(~hours.eq(24), 0)
                index = pd.DatetimeIndex(
                    date_values
                    + pd.to_timedelta(day_add, unit="D")
                    + pd.to_timedelta(normalized_hours, unit="h")
                    + pd.to_timedelta(minutes, unit="m")
                    + pd.to_timedelta(seconds, unit="s")
                )
            else:
                combined = (
                    frame[date_col].astype(str).str.strip()
                    + " "
                    + time_values
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
    promoted = _promote_numeric_columns(frame)
    return list(promoted.select_dtypes(include=[np.number]).columns)


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

            frame = _promote_numeric_columns(frame)
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
