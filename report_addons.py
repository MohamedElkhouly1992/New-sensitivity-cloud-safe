from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Weather upload utilities
# -----------------------------
def _read_csv_fallback(file_or_path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "ISO-8859-1"]
    last_error = None
    for enc in encodings:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception as exc:  # pragma: no cover - keeps upload robust
            last_error = exc
    raise ValueError(f"Could not read CSV file. Last error: {last_error}")



def _read_designbuilder_grid_csv(file_or_path) -> pd.DataFrame:
    """Read a DesignBuilder model-data-grid CSV export.

    DesignBuilder grid exports usually contain three metadata rows followed by the
    actual header row and then one descriptive/unit row. This reader handles that
    format and also falls back to normal CSV/Excel-style parsing.
    """
    last_error = None
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252", "ISO-8859-1"]:
        for skip in [3, 0, 1, 2]:
            try:
                if hasattr(file_or_path, "seek"):
                    file_or_path.seek(0)
                df = pd.read_csv(file_or_path, encoding=enc, skiprows=skip, engine="python")
                df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
                df = df.dropna(how="all")
                if df.shape[1] >= 3:
                    # Drop DesignBuilder descriptive row such as "Model Object Identifier".
                    first = " ".join([str(x) for x in df.iloc[0].tolist()[:4]]).lower() if len(df) else ""
                    if "model object identifier" in first or "inherited zone handle" in first:
                        df = df.iloc[1:].copy()
                    return df.reset_index(drop=True)
            except Exception as exc:
                last_error = exc
    raise ValueError(f"Could not read DesignBuilder grid file. Last error: {last_error}")


def _to_float_series(values, default=np.nan) -> pd.Series:
    s = pd.Series(values).astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _db_pick(df: pd.DataFrame, names: list[str]):
    if df is None or df.empty:
        return None
    return _infer_col(df.columns, names)


def _db_zone_key(df: pd.DataFrame) -> pd.Series:
    title = _db_pick(df, ["Title", "Zone", "Identity - Identity - Name"])
    zone = _db_pick(df, ["Zone"])
    if title is not None:
        return df[title].astype(str).str.strip()
    if zone is not None:
        return df[zone].astype(str).str.strip()
    return pd.Series([f"Zone_{i+1}" for i in range(len(df))], index=df.index)


def build_designbuilder_zone_table_from_uploads(uploaded_files) -> pd.DataFrame:
    """Build a solver zone table from one or more DesignBuilder zone CSV exports.

    Expected useful exports include zones.csv, internal loads.csv, set points.csv,
    zone lighting.csv, zone hvac.csv, and mechanical/natural ventilation grids.
    The returned table is compatible with aggregate_zone_occupancy() and the
    native zone-load engine.
    """
    if not uploaded_files:
        return pd.DataFrame()
    frames: dict[str, pd.DataFrame] = {}
    for uf in uploaded_files:
        name = str(getattr(uf, "name", "uploaded")).lower()
        try:
            df = _read_designbuilder_grid_csv(uf)
        except Exception:
            continue
        frames[name] = df

    # choose the richest base table with FloorArea if present
    base_name = None
    for key in frames:
        low = key.lower()
        if "internal" in low:
            base_name = key; break
    if base_name is None:
        for key, df in frames.items():
            if _db_pick(df, ["FloorArea", "Floor Area", "Area"]) is not None:
                base_name = key; break
    if base_name is None:
        for key in frames:
            if "zones" in key.lower():
                base_name = key; break
    if base_name is None:
        base_name = next(iter(frames))

    base = frames[base_name].copy()
    out = pd.DataFrame(index=base.index)
    out["zone_name"] = _db_zone_key(base)
    block_col = _db_pick(base, ["Block"])
    out["block"] = base[block_col].astype(str) if block_col is not None else ""
    ztype_col = _db_pick(base, ["ZoneType", "ActivityTemplateId", "Zone Type"])
    out["zone_type"] = base[ztype_col].astype(str).str.split("|").str[0] if ztype_col is not None else "Custom"
    area_col = _db_pick(base, ["FloorArea", "Floor Area", "Area"])
    out["area_m2"] = _to_float_series(base[area_col], default=0.0) if area_col is not None else 0.0
    occ_col = _db_pick(base, ["OccupancyValue", "Occupancy", "People"])
    out["occ_density"] = _to_float_series(base[occ_col], default=np.nan) if occ_col is not None else np.nan

    # Merge additional DesignBuilder grids by zone_name.
    def merge_from_frame(df: pd.DataFrame, mappings: dict[str, tuple[list[str], float | None]]):
        nonlocal out
        if df is None or df.empty:
            return
        temp = pd.DataFrame({"zone_name": _db_zone_key(df)})
        for out_col, (candidates, default) in mappings.items():
            col = _db_pick(df, candidates)
            if col is not None:
                temp[out_col] = _to_float_series(df[col], default=np.nan if default is None else default)
        if len(temp.columns) > 1:
            temp = temp.drop_duplicates("zone_name")
            out = out.merge(temp, on="zone_name", how="left", suffixes=("", "_dbnew"))
            for c in list(out.columns):
                if c.endswith("_dbnew"):
                    base_c = c[:-6]
                    out[base_c] = out[base_c].combine_first(out[c]) if base_c in out.columns else out[c]
                    out = out.drop(columns=[c])

    for key, df in frames.items():
        low = key.lower()
        if "set" in low:
            merge_from_frame(df, {
                "heating_setpoint_c": (["HeatingDesignSetPointTemperature", "Heating Setpoint"], 22.0),
                "cooling_setpoint_c": (["ActiveCoolingDesignSetPointTemperature", "Cooling Setpoint"], 24.0),
            })
        if "lighting" in low:
            # DesignBuilder gives W/m² per 100 lux and target illuminance. Convert approximate W/m².
            temp = pd.DataFrame({"zone_name": _db_zone_key(df)})
            lp_col = _db_pick(df, ["FluorescentLightingWattsPerM2Per100Lux"])
            ill_col = _db_pick(df, ["TargetIlluminance1", "TargetIlluminance2"])
            if lp_col is not None:
                lp = _to_float_series(df[lp_col], default=np.nan)
                if ill_col is not None:
                    ill = _to_float_series(df[ill_col], default=300.0)
                    temp["lighting_w_m2"] = lp * ill / 100.0
                else:
                    temp["lighting_w_m2"] = lp
                temp = temp.drop_duplicates("zone_name")
                out = out.merge(temp, on="zone_name", how="left", suffixes=("", "_light"))
                if "lighting_w_m2_light" in out.columns:
                    out["lighting_w_m2"] = out.get("lighting_w_m2").combine_first(out["lighting_w_m2_light"]) if "lighting_w_m2" in out.columns else out["lighting_w_m2_light"]
                    out = out.drop(columns=["lighting_w_m2_light"])
        if "hvac" in low or "mechanical" in low:
            merge_from_frame(df, {
                "mechanical_ventilation_value": (["MechanicalVentilationValue"], None),
                "min_fresh_air_l_s_person": (["MinFreshAir"], None),
                "min_fresh_air_l_s_m2": (["MinFreshAirPerArea"], None),
            })
        if "internal" in low:
            merge_from_frame(df, {
                "equipment_w_m2": (["EquipmentValue"], None),
                "computers_w_m2": (["ComputersValue"], None),
            })

    # Fill defaults and schedule proxies.
    out["area_m2"] = pd.to_numeric(out["area_m2"], errors="coerce").fillna(0.0)
    out = out[out["area_m2"] > 0].copy()
    out["occ_density"] = pd.to_numeric(out["occ_density"], errors="coerce").fillna(0.08)
    out["zone_type"] = out["zone_type"].replace({"nan": "Custom", "": "Custom"}).fillna("Custom")
    for c, v in {
        "lighting_w_m2": 10.0,
        "equipment_w_m2": 8.0,
        "computers_w_m2": 0.0,
        "heating_setpoint_c": 22.0,
        "cooling_setpoint_c": 24.0,
        "term_factor": 0.85,
        "break_factor": 0.35,
        "summer_factor": 0.50,
    }.items():
        if c not in out.columns:
            out[c] = v
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(v)

    # Infer schedule factors from DesignBuilder office/university activity strings when present.
    zlow = out["zone_type"].astype(str).str.lower()
    out.loc[zlow.str.contains("office|lecture|class|generic", regex=True), ["term_factor", "break_factor", "summer_factor"]] = [0.85, 0.35, 0.50]
    out.loc[zlow.str.contains("corridor|toilet|service|stair", regex=True), ["term_factor", "break_factor", "summer_factor"]] = [0.55, 0.40, 0.40]

    keep = [
        "zone_name", "zone_type", "block", "area_m2", "occ_density",
        "lighting_w_m2", "equipment_w_m2", "computers_w_m2",
        "heating_setpoint_c", "cooling_setpoint_c",
        "mechanical_ventilation_value", "min_fresh_air_l_s_person", "min_fresh_air_l_s_m2",
        "term_factor", "break_factor", "summer_factor",
    ]
    keep = [c for c in keep if c in out.columns]
    return out[keep].reset_index(drop=True)


def _infer_col(cols, candidates):
    normalized = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    for c in cols:
        low = str(c).strip().lower()
        for cand in candidates:
            if cand.strip().lower() in low:
                return c
    return None


def normalize_weather_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return engine-format 365-row daily weather dataframe."""
    if df is None or df.empty:
        raise ValueError("Weather dataframe is empty.")
    work = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    if {"day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"}.issubset(work.columns):
        out = work[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]].copy()
    else:
        date_col = _infer_col(work.columns, ["Date/Time", "date", "datetime", "timestamp", "time"])
        doy_col = _infer_col(work.columns, ["day_of_year", "doy", "day"])
        temp_col = _infer_col(work.columns, ["T_mean_C", "T_amb_C", "Outdoor Dry-Bulb Temperature", "DryBulb", "dry-bulb", "temperature", "temp"])
        tmax_col = _infer_col(work.columns, ["T_max_C", "max temperature", "temperature max", "Tmax"])
        rh_col = _infer_col(work.columns, ["RH_mean_pct", "RH_pct", "Relative Humidity", "humidity", "rh"])
        ghi_col = _infer_col(work.columns, ["GHI_mean_Wm2", "GHI_Wm2", "Global Solar Radiation", "Global Horizontal Solar", "solar", "ghi"])
        direct_col = _infer_col(work.columns, ["Direct Normal Solar", "Direct Solar", "DNI"])
        diffuse_col = _infer_col(work.columns, ["Diffuse Horizontal Solar", "Diffuse Solar", "DHI"])
        if temp_col is None:
            raise ValueError("Weather CSV must contain outdoor temperature column.")
        out = pd.DataFrame()
        if date_col is not None:
            dates = pd.to_datetime(work[date_col], errors="coerce")
            out["day_of_year"] = dates.dt.dayofyear
        elif doy_col is not None:
            out["day_of_year"] = pd.to_numeric(work[doy_col], errors="coerce")
        else:
            out["day_of_year"] = np.arange(1, len(work) + 1)
        out["T_mean_C"] = pd.to_numeric(work[temp_col], errors="coerce")
        out["T_max_C"] = pd.to_numeric(work[tmax_col], errors="coerce") if tmax_col is not None else out["T_mean_C"] + 5.0
        out["RH_mean_pct"] = pd.to_numeric(work[rh_col], errors="coerce") if rh_col is not None else 60.0
        if ghi_col is not None:
            out["GHI_mean_Wm2"] = pd.to_numeric(work[ghi_col], errors="coerce")
        else:
            solar = pd.Series(np.zeros(len(work)), index=work.index, dtype=float)
            if direct_col is not None:
                solar += pd.to_numeric(work[direct_col], errors="coerce").fillna(0.0)
            if diffuse_col is not None:
                solar += pd.to_numeric(work[diffuse_col], errors="coerce").fillna(0.0)
            out["GHI_mean_Wm2"] = solar
    for c in ["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["day_of_year", "T_mean_C"]).copy()
    out["day_of_year"] = out["day_of_year"].astype(int)
    out = out[(out["day_of_year"] >= 1) & (out["day_of_year"] <= 366)].copy()
    out.loc[out["day_of_year"] > 365, "day_of_year"] = 365
    out = out.groupby("day_of_year", as_index=False).agg(
        T_mean_C=("T_mean_C", "mean"),
        T_max_C=("T_max_C", "mean"),
        RH_mean_pct=("RH_mean_pct", "mean"),
        GHI_mean_Wm2=("GHI_mean_Wm2", "mean"),
    ).sort_values("day_of_year")
    out = out.set_index("day_of_year").reindex(range(1, 366))
    out[["T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]] = out[["T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]].interpolate(limit_direction="both").ffill().bfill()
    out = out.reset_index().rename(columns={"index": "day_of_year"})
    if out.isna().any().any():
        raise ValueError("Weather normalization failed; missing values remain after interpolation.")
    return out[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]]


def read_epw_upload(uploaded_file) -> pd.DataFrame:
    """Return native hourly EPW records for the engine to resample to the selected time-step."""
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    rows = []
    for line in content[8:]:
        parts = line.split(",")
        if len(parts) < 14:
            continue
        try:
            month = int(float(parts[1])); day = int(float(parts[2])); hour = int(float(parts[3]))
            if month == 2 and day == 29:
                continue
            dry = float(parts[6]); rh = float(parts[8]); ghi = max(float(parts[13]), 0.0)
            ts = pd.Timestamp(year=2001, month=month, day=day, hour=max(0, min(hour - 1, 23)))
            rows.append({"Date/Time": ts, "T_mean_C": dry, "T_max_C": dry, "RH_mean_pct": rh, "GHI_mean_Wm2": ghi})
        except Exception:
            continue
    if not rows:
        raise ValueError("No valid hourly rows parsed from EPW upload.")
    return pd.DataFrame(rows)


def standardize_weather_upload_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return timestamped CSV rows when available; otherwise return 365 daily rows.

    The engine performs final normalization/resampling according to the selected time-step.
    This function intentionally avoids forcing all uploads into daily records so hourly CSV/EPW
    files can drive true sub-daily calculations.
    """
    if df is None or df.empty:
        raise ValueError("Weather dataframe is empty.")
    work = df[[c for c in df.columns if not str(c).startswith("Unnamed")]].copy()
    date_col = _infer_col(work.columns, ["Date/Time", "date", "datetime", "timestamp", "time"])
    temp_col = _infer_col(work.columns, ["T_mean_C", "T_amb_C", "Outdoor Dry-Bulb Temperature", "DryBulb", "dry-bulb", "temperature", "temp"])
    tmax_col = _infer_col(work.columns, ["T_max_C", "max temperature", "temperature max", "Tmax"])
    rh_col = _infer_col(work.columns, ["RH_mean_pct", "RH_pct", "Relative Humidity", "humidity", "rh"])
    ghi_col = _infer_col(work.columns, ["GHI_mean_Wm2", "GHI_Wm2", "Global Solar Radiation", "Global Horizontal Solar", "solar", "ghi"])
    direct_col = _infer_col(work.columns, ["Direct Normal Solar", "Direct Solar", "DNI"])
    diffuse_col = _infer_col(work.columns, ["Diffuse Horizontal Solar", "Diffuse Solar", "DHI"])
    if temp_col is None:
        raise ValueError("Weather CSV must contain outdoor temperature column.")
    out = pd.DataFrame()
    if date_col is not None:
        dates = pd.to_datetime(work[date_col], errors="coerce")
        if dates.notna().sum() >= max(3, int(0.5 * len(work))):
            out["Date/Time"] = dates
    if "Date/Time" not in out.columns:
        doy_col = _infer_col(work.columns, ["day_of_year", "doy", "day"])
        if doy_col is not None:
            out["day_of_year"] = pd.to_numeric(work[doy_col], errors="coerce")
        else:
            out["day_of_year"] = np.arange(1, len(work) + 1)
    out["T_mean_C"] = pd.to_numeric(work[temp_col], errors="coerce")
    out["T_max_C"] = pd.to_numeric(work[tmax_col], errors="coerce") if tmax_col is not None else out["T_mean_C"]
    out["RH_mean_pct"] = pd.to_numeric(work[rh_col], errors="coerce") if rh_col is not None else 60.0
    if ghi_col is not None:
        out["GHI_mean_Wm2"] = pd.to_numeric(work[ghi_col], errors="coerce")
    else:
        solar = pd.Series(np.zeros(len(work)), index=work.index, dtype=float)
        if direct_col is not None:
            solar += pd.to_numeric(work[direct_col], errors="coerce").fillna(0.0)
        if diffuse_col is not None:
            solar += pd.to_numeric(work[diffuse_col], errors="coerce").fillna(0.0)
        out["GHI_mean_Wm2"] = solar
    out = out.dropna(subset=["T_mean_C"]).copy()
    if "Date/Time" not in out.columns:
        return normalize_weather_df(out)
    out = out.dropna(subset=["Date/Time"]).sort_values("Date/Time")
    return out.reset_index(drop=True)


def read_weather_upload(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "").lower()
    if name.endswith(".epw"):
        return read_epw_upload(uploaded_file)
    return standardize_weather_upload_df(_read_csv_fallback(uploaded_file))


# Detailed output sheets
# -----------------------------
def find_result_paths(folder: str | Path) -> Dict[str, Path]:
    folder = Path(folder)
    summary_candidates = [
        folder / "baseline_scenario_summary.csv",
        folder / "three_axis_summary.csv",
        folder / "matrix_summary.csv",
        folder / "one_axis_strategy_summary.csv",
        folder / "one_axis_severity_summary.csv",
        folder / "baseline_no_degradation_summary.csv",
    ]
    annual_candidates = [
        folder / "annual_baseline_scenario.csv",
        folder / "annual_three_axis.csv",
        folder / "annual_matrix.csv",
        folder / "annual_one_axis_strategy.csv",
        folder / "annual_one_axis_severity.csv",
        folder / "baseline_no_degradation_annual.csv",
    ]
    daily_candidates = [
        folder / "baseline_scenario_ml_dataset.csv",
        folder / "three_axis_ml_dataset.csv",
        folder / "matrix_ml_dataset.csv",
        folder / "one_axis_strategy_ml_dataset.csv",
        folder / "one_axis_severity_ml_dataset.csv",
        folder / "baseline_no_degradation_daily.csv",
    ]
    def first_existing(cands):
        for p in cands:
            if p.exists():
                return p
        return cands[0]
    return {"summary": first_existing(summary_candidates), "annual": first_existing(annual_candidates), "daily": first_existing(daily_candidates)}


def _read_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_kpi_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "Total Energy MWh", "Total Cost USD", "Total CO2 tonne", "Mean COP", "Mean Degradation Index", "Mean Comfort Deviation C", "Occupied Discomfort Days"] if c in summary_df.columns]
    return summary_df[cols].copy()


def build_fuel_breakdown(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    df = daily_df.copy()
    mode = df.get("mode", pd.Series("unknown", index=df.index)).astype(str)
    energy_col = "energy_kwh_period" if "energy_kwh_period" in df.columns else "energy_kwh_day"
    co2_col = "co2_kg_period" if "co2_kg_period" in df.columns else "co2_kg_day"
    cost_col = "cost_usd_period" if "cost_usd_period" in df.columns else "cost_usd_day"

    for col in ["thermal_hvac_kwh_period", "fan_kwh_period", "pump_kwh_period", "auxiliary_kwh_period"]:
        if col not in df.columns:
            df[col] = 0.0
    if "cooling_hvac_kwh_period" in df.columns or "heating_hvac_kwh_period" in df.columns:
        df["cooling_thermal_hvac_kwh"] = pd.to_numeric(df.get("cooling_hvac_kwh_period", 0.0), errors="coerce").fillna(0.0)
        df["heating_thermal_hvac_kwh"] = pd.to_numeric(df.get("heating_hvac_kwh_period", 0.0), errors="coerce").fillna(0.0)
    else:
        df["cooling_thermal_hvac_kwh"] = np.where(mode.eq("cooling"), df["thermal_hvac_kwh_period"], 0.0)
        df["heating_thermal_hvac_kwh"] = np.where(mode.eq("heating"), df["thermal_hvac_kwh_period"], 0.0)
    keys = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year"] if c in df.columns]
    if not keys:
        keys = ["year"] if "year" in df.columns else []
    out = df.groupby(keys, as_index=False).agg(
        total_energy_kwh=(energy_col, "sum"),
        thermal_hvac_kwh=("thermal_hvac_kwh_period", "sum"),
        cooling_thermal_hvac_kwh=("cooling_thermal_hvac_kwh", "sum"),
        heating_thermal_hvac_kwh=("heating_thermal_hvac_kwh", "sum"),
        fan_kwh=("fan_kwh_period", "sum"),
        pump_kwh=("pump_kwh_period", "sum"),
        auxiliary_kwh=("auxiliary_kwh_period", "sum"),
        co2_kg=(co2_col, "sum"),
        cost_usd=(cost_col, "sum"),
    )
    out["hvac_balance_check_kwh"] = out[["thermal_hvac_kwh", "fan_kwh", "pump_kwh", "auxiliary_kwh"]].sum(axis=1)
    out["balance_error_kwh"] = out["total_energy_kwh"] - out["hvac_balance_check_kwh"]
    return out


def build_comfort_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "day", "year", "day_of_year", "occ", "T_sp_C", "T_amb_C", "RH_mean_pct", "comfort_dev_C", "occupied_discomfort_flag"] if c in daily_df.columns]
    return daily_df[cols].copy()


def build_site_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["scenario_combo_3axis", "day", "year", "day_of_year", "T_amb_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "occ"] if c in daily_df.columns]
    return daily_df[cols].drop_duplicates().copy()


def build_internal_gains(daily_df: pd.DataFrame, bldg=None) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame()
    df = daily_df.copy()
    area = getattr(bldg, "conditioned_area_m2", float(df.get("area_m2", pd.Series([0])).iloc[0] if "area_m2" in df else 0))
    occ_density = getattr(bldg, "occupancy_density_p_m2", 0.0)
    lighting = getattr(bldg, "lighting_w_m2", 0.0)
    equipment = getattr(bldg, "equipment_w_m2", 0.0)
    sensible = getattr(bldg, "sensible_w_per_person", 0.0)
    occ = df.get("occ", 0.0)
    df["people_gain_kw"] = area * occ_density * sensible * occ / 1000.0
    df["lighting_gain_kw"] = area * lighting * occ / 1000.0
    df["equipment_gain_kw"] = area * equipment * occ / 1000.0
    df["total_internal_gain_kw"] = df["people_gain_kw"] + df["lighting_gain_kw"] + df["equipment_gain_kw"]
    cols = [c for c in ["scenario_combo_3axis", "day", "year", "day_of_year", "occ", "people_gain_kw", "lighting_gain_kw", "equipment_gain_kw", "total_internal_gain_kw"] if c in df.columns]
    return df[cols].copy()


def build_benchmark_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "Total Energy MWh" not in summary_df.columns:
        return pd.DataFrame()
    df = summary_df.copy()
    base = None
    if "scenario_combo_3axis" in df.columns:
        baseline_rows = df[df["scenario_combo_3axis"].astype(str).str.contains("BASELINE", case=False, na=False)]
        if not baseline_rows.empty:
            base = float(baseline_rows.iloc[0]["Total Energy MWh"])
    if base is None:
        base = float(df["Total Energy MWh"].iloc[0])
    df["energy_delta_MWh"] = df["Total Energy MWh"] - base
    df["energy_delta_pct"] = 100.0 * df["energy_delta_MWh"] / max(abs(base), 1e-9)
    if "Mean Degradation Index" in df.columns:
        df["degradation_delta"] = df["Mean Degradation Index"] - float(df["Mean Degradation Index"].iloc[0])
    keep = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "Total Energy MWh", "energy_delta_MWh", "energy_delta_pct", "Mean Degradation Index", "degradation_delta", "Mean Comfort Deviation C", "Total CO2 tonne"] if c in df.columns]
    return df[keep]


def build_zone_analysis(daily_df: pd.DataFrame, zone_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if daily_df.empty or zone_df is None or len(zone_df) == 0:
        return pd.DataFrame()
    z = zone_df.copy()
    required = {"zone_name", "area_m2", "occ_density"}
    if not required.issubset(z.columns):
        return pd.DataFrame()
    z["area_m2"] = pd.to_numeric(z["area_m2"], errors="coerce").fillna(0.0)
    z["occ_density"] = pd.to_numeric(z["occ_density"], errors="coerce").fillna(0.0)
    weights = z["area_m2"] * np.maximum(z["occ_density"], 0.01)
    if float(weights.sum()) <= 0:
        weights = z["area_m2"]
    if float(weights.sum()) <= 0:
        weights = pd.Series(np.ones(len(z)), index=z.index)
    weights = weights / weights.sum()
    summaries = []
    keys = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year"] if c in daily_df.columns]
    grouped = daily_df.groupby(keys, dropna=False) if keys else [((), daily_df)]
    for group_key, grp in grouped:
        base = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,))) if keys else {}
        energy = float(grp.get("energy_kwh_day", pd.Series([0])).sum())
        co2 = float(grp.get("co2_kg_day", pd.Series([0])).sum())
        comfort = float(grp.get("comfort_dev_C", pd.Series([0])).mean())
        deg = float(grp.get("delta", pd.Series([0])).mean())
        for idx, row in z.iterrows():
            rec = dict(base)
            mult = float(weights.loc[idx])
            rec.update({
                "zone_name": row.get("zone_name", f"Zone_{idx+1}"),
                "zone_type": row.get("zone_type", "Custom"),
                "zone_area_m2": float(row.get("area_m2", 0)),
                "zone_occ_density": float(row.get("occ_density", 0)),
                "zone_energy_kwh": energy * mult,
                "zone_co2_kg": co2 * mult,
                "zone_comfort_dev_C": comfort,
                "zone_degradation_index": deg,
            })
            summaries.append(rec)
    return pd.DataFrame(summaries)


def build_validation_template(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        scenarios = ["Example_Scenario"]
    else:
        scenarios = summary_df.get("scenario_combo_3axis", pd.Series(range(len(summary_df)))).astype(str).tolist()
    return pd.DataFrame({
        "scenario_combo_3axis": scenarios,
        "reference_energy_MWh": np.nan,
        "reference_co2_tonne": np.nan,
        "reference_comfort_dev_C": np.nan,
        "reference_degradation_index": np.nan,
        "source_note": "DesignBuilder / EnergyPlus / measured / published reference",
    })


def load_validation_file(file_or_path) -> pd.DataFrame:
    return _read_csv_fallback(file_or_path)


def build_validation_comparison(summary_df: pd.DataFrame, validation_df: pd.DataFrame, source_name: str = "validation") -> pd.DataFrame:
    if summary_df.empty or validation_df.empty:
        return pd.DataFrame()
    val = validation_df.copy()
    scen_val = _infer_col(val.columns, ["scenario_combo_3axis", "Scenario Key", "scenario", "case", "strategy"])
    model = summary_df.copy()
    scen_model = "scenario_combo_3axis" if "scenario_combo_3axis" in model.columns else None
    def pick(df, names):
        return _infer_col(df.columns, names)
    energy_ref = pick(val, ["reference_energy_MWh", "Total Energy MWh", "Energy MWh", "energy", "Energy Consumption (kWh)"])
    co2_ref = pick(val, ["reference_co2_tonne", "Total CO2 tonne", "CO2", "Carbon Footprint"])
    comfort_ref = pick(val, ["reference_comfort_dev_C", "Mean Comfort Deviation C", "comfort"])
    if scen_val and scen_model:
        merged = model.merge(val, left_on=scen_model, right_on=scen_val, how="left", suffixes=("_model", "_ref"))
    else:
        merged = pd.concat([model.reset_index(drop=True), val.reset_index(drop=True)], axis=1)
    rows = []
    for _, r in merged.iterrows():
        rec = {"source": source_name, "scenario_combo_3axis": r.get("scenario_combo_3axis", r.get(scen_val, "case"))}
        if energy_ref and "Total Energy MWh" in merged.columns:
            ref = pd.to_numeric(pd.Series([r.get(energy_ref)]), errors="coerce").iloc[0]
            mod = pd.to_numeric(pd.Series([r.get("Total Energy MWh")]), errors="coerce").iloc[0]
            rec.update({"model_energy_MWh": mod, "reference_energy_MWh": ref, "energy_error_pct": 100.0 * (mod - ref) / max(abs(ref), 1e-9) if pd.notna(ref) else np.nan})
        if co2_ref and "Total CO2 tonne" in merged.columns:
            ref = pd.to_numeric(pd.Series([r.get(co2_ref)]), errors="coerce").iloc[0]
            mod = pd.to_numeric(pd.Series([r.get("Total CO2 tonne")]), errors="coerce").iloc[0]
            rec.update({"model_co2_tonne": mod, "reference_co2_tonne": ref, "co2_error_pct": 100.0 * (mod - ref) / max(abs(ref), 1e-9) if pd.notna(ref) else np.nan})
        if comfort_ref and "Mean Comfort Deviation C" in merged.columns:
            ref = pd.to_numeric(pd.Series([r.get(comfort_ref)]), errors="coerce").iloc[0]
            mod = pd.to_numeric(pd.Series([r.get("Mean Comfort Deviation C")]), errors="coerce").iloc[0]
            rec.update({"model_comfort_dev_C": mod, "reference_comfort_dev_C": ref, "comfort_error_C": mod - ref if pd.notna(ref) else np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


def build_detailed_tables(folder: str | Path, bldg=None, cfg=None, zone_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    paths = find_result_paths(folder)
    summary_df = _read_if_exists(paths["summary"])
    annual_df = _read_if_exists(paths["annual"])
    daily_df = _read_if_exists(paths["daily"])
    return {
        "kpi_summary": build_kpi_summary(summary_df),
        "fuel_breakdown": build_fuel_breakdown(daily_df),
        "comfort": build_comfort_table(daily_df),
        "site_data": build_site_data(daily_df),
        "internal_gains": build_internal_gains(daily_df, bldg=bldg),
        "validation_template": build_validation_template(summary_df),
        "benchmark_summary": build_benchmark_summary(summary_df),
        "zone_analysis": build_zone_analysis(daily_df, zone_df),
        "summary_copy": summary_df,
        "annual_copy": annual_df,
    }


def save_detailed_outputs(folder: str | Path, tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            path = folder / f"{name}.csv"
            df.to_csv(path, index=False)
            saved[name] = str(path)
    xlsx = folder / "detailed_outputs.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.head(50000).to_excel(writer, sheet_name=name[:31], index=False)
    saved["detailed_outputs_excel"] = str(xlsx)
    return saved


def create_zip_from_folder(folder: str | Path) -> Path:
    folder = Path(folder)
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in folder.rglob("*"):
            if p.is_file() and p != zip_path:
                zf.write(p, arcname=str(p.relative_to(folder)))
    return zip_path


# -----------------------------
# Setup JSON helpers
# -----------------------------
def setup_to_json_bytes(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def setup_from_upload(uploaded_file) -> Dict[str, Any]:
    raw = uploaded_file.getvalue().decode("utf-8")
    return json.loads(raw)


# -----------------------------------------------------------------------------
# Publication-plus diagnostic/refinement modules
# -----------------------------------------------------------------------------
def saturation_vapor_pressure_kpa(T_c: float) -> float:
    """Tetens saturation vapor pressure, kPa."""
    T_c = float(T_c)
    return 0.61078 * np.exp((17.2694 * T_c) / (T_c + 237.3))


def humidity_ratio_kgkg(T_c: float, RH_pct: float, pressure_kpa: float = 101.325) -> float:
    """Approximate humidity ratio from dry-bulb temperature and relative humidity."""
    rh = np.clip(float(RH_pct) / 100.0, 0.0, 1.0)
    p_ws = saturation_vapor_pressure_kpa(float(T_c))
    p_w = min(rh * p_ws, 0.99 * pressure_kpa)
    return float(0.62198 * p_w / max(pressure_kpa - p_w, 1e-9))


def _period_energy_col(df: pd.DataFrame) -> str:
    return "energy_kwh_period" if "energy_kwh_period" in df.columns else "energy_kwh_day"


def _duration_col_value(df: pd.DataFrame) -> pd.Series:
    if "duration_hours" in df.columns:
        return pd.to_numeric(df["duration_hours"], errors="coerce").fillna(24.0)
    if "time_step_hours" in df.columns:
        return pd.to_numeric(df["time_step_hours"], errors="coerce").fillna(24.0)
    return pd.Series(np.full(len(df), 24.0), index=df.index)


def build_heat_exchanger_diagnostics(
    daily_df: pd.DataFrame,
    bldg=None,
    cfg=None,
    air_inlet_mode: str = "mixed_air_estimate",
    fixed_air_inlet_c: float = 26.0,
    chilled_water_in_c: float = 7.0,
    hot_water_in_c: float = 60.0,
    water_flow_m3h: float = 0.0,
    water_dp_clean_kpa: float = 45.0,
    pump_efficiency: float = 0.65,
    ua_clean_kw_k: float = 120.0,
    ua_loss_factor: float = 0.30,
    air_fouling_dp_factor: float = 0.75,
    water_fouling_dp_factor: float = 0.35,
    lmtd_correction_factor: float = 0.90,
) -> pd.DataFrame:
    """Build heat-exchanger diagnostics from model time-step output.

    This is a diagnostic/refinement layer. It does not replace the core engine,
    but it links load, fouling/degradation, pressure drop, outlet temperatures,
    UA loss, and pump/fan implications for publication reporting.
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    df = daily_df.copy()
    area = float(getattr(bldg, "conditioned_area_m2", df.get("area_m2", pd.Series([5000.0])).iloc[0] if "area_m2" in df else 5000.0))
    airflow_m3h_m2 = float(getattr(bldg, "airflow_m3h_m2", 4.0))
    q_air_nom = area * airflow_m3h_m2
    rho_air = 1.20
    cp_air = 1.006  # kJ/kg.K
    rho_water = 997.0
    cp_water = 4.186  # kJ/kg.K
    dp_clean = float(getattr(cfg, "DP_CLEAN", 150.0))
    dp_max = float(getattr(cfg, "DP_MAX", 450.0))

    af = pd.to_numeric(df.get("alpha_flow", 1.0), errors="coerce").fillna(1.0).clip(lower=0.05)
    occ = pd.to_numeric(df.get("occ", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0)
    delta = pd.to_numeric(df.get("delta", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    t_amb = pd.to_numeric(df.get("T_amb_C", 25.0), errors="coerce").fillna(25.0)
    t_sp = pd.to_numeric(df.get("T_sp_C", getattr(cfg, "T_SET", 23.0) if cfg is not None else 23.0), errors="coerce").fillna(23.0)
    q = pd.to_numeric(df.get("Q_HVAC_kw", 0.0), errors="coerce").fillna(0.0)
    q_cool = pd.to_numeric(df.get("Q_cool_kw", 0.0), errors="coerce").fillna(0.0)
    q_heat = pd.to_numeric(df.get("Q_heat_kw", 0.0), errors="coerce").fillna(0.0)
    mode = df.get("mode", pd.Series(np.where(q_cool >= q_heat, "cooling", "heating"), index=df.index)).astype(str)
    airflow_m3h = q_air_nom * af
    m_air = airflow_m3h / 3600.0 * rho_air

    if air_inlet_mode == "fixed":
        t_air_in = pd.Series(np.full(len(df), float(fixed_air_inlet_c)), index=df.index)
    elif air_inlet_mode == "ambient":
        t_air_in = t_amb
    else:
        # Mixed/return-air estimate: return air near setpoint, with weather and occupancy influence.
        t_air_in = t_sp + 0.25 * (t_amb - t_sp) + 0.80 * occ

    delta_t_air = q / np.maximum(m_air * cp_air, 1e-9)
    t_air_out = np.where(mode.eq("cooling"), t_air_in - delta_t_air, t_air_in + delta_t_air)

    # Estimate water flow if the user leaves it at zero. Uses a 5 K chilled-water delta-T or 10 K hot-water delta-T.
    if water_flow_m3h and float(water_flow_m3h) > 0:
        water_flow = pd.Series(np.full(len(df), float(water_flow_m3h)), index=df.index)
    else:
        design_dt = np.where(mode.eq("cooling"), 5.0, 10.0)
        water_flow = (q / np.maximum(rho_water * cp_water * design_dt / 3600.0, 1e-9)).clip(lower=0.0)
    m_water = water_flow / 3600.0 * rho_water
    t_water_in = np.where(mode.eq("cooling"), chilled_water_in_c, hot_water_in_c)
    water_dt = q / np.maximum(m_water * cp_water, 1e-9)
    t_water_out = np.where(mode.eq("cooling"), t_water_in + water_dt, t_water_in - water_dt)

    dP_air = np.minimum(dp_clean * (af ** 2) * (1.0 + air_fouling_dp_factor * delta), dp_max)
    flow_ratio = water_flow / max(float(water_flow.mean()) if float(water_flow.mean()) > 0 else 1.0, 1e-9)
    dP_water_kpa = water_dp_clean_kpa * (flow_ratio ** 2) * (1.0 + water_fouling_dp_factor * delta)
    pump_kw_detailed = (water_flow / 3600.0) * (dP_water_kpa * 1000.0) / max(float(pump_efficiency), 1e-6) / 1000.0

    ua_effective = float(ua_clean_kw_k) * (1.0 - float(ua_loss_factor) * delta.clip(upper=1.0))
    ua_effective = np.maximum(ua_effective, 1e-6)

    dt1 = np.where(mode.eq("cooling"), t_air_in - t_water_out, t_water_in - t_air_out)
    dt2 = np.where(mode.eq("cooling"), t_air_out - t_water_in, t_water_out - t_air_in)
    dt1 = np.maximum(np.abs(dt1), 1e-6)
    dt2 = np.maximum(np.abs(dt2), 1e-6)
    lmtd = np.where(np.abs(dt1 - dt2) < 1e-6, dt1, (dt1 - dt2) / np.log(np.maximum(dt1 / dt2, 1e-9)))
    q_ua_kw = ua_effective * lmtd * float(lmtd_correction_factor)
    effectiveness = q / np.maximum(q_ua_kw, 1e-9)

    out = df[[c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year", "day", "day_of_year", "hour_of_day", "mode"] if c in df.columns]].copy()
    out["Q_HVAC_kw"] = q
    out["airflow_m3h"] = airflow_m3h
    out["air_mass_flow_kg_s"] = m_air
    out["T_air_in_C"] = t_air_in
    out["T_air_out_C"] = t_air_out
    out["water_flow_m3h"] = water_flow
    out["T_water_in_C"] = t_water_in
    out["T_water_out_C"] = t_water_out
    out["dP_air_Pa"] = dP_air
    out["dP_water_kPa"] = dP_water_kpa
    out["UA_clean_kW_K"] = float(ua_clean_kw_k)
    out["UA_effective_kW_K"] = ua_effective
    out["LMTD_C"] = lmtd
    out["effectiveness_ratio"] = effectiveness
    out["detailed_pump_kw"] = pump_kw_detailed
    out["diagnostic_note"] = "post-processing HX diagnostic; optionally use for fan/pump/COP correction in interpretation"
    return out


def build_part_load_curve_analysis(
    daily_df: pd.DataFrame,
    cfg=None,
    curve_type: str = "quadratic",
    coeff_a: float = 0.85,
    coeff_b: float = 0.25,
    coeff_c: float = -0.10,
    coeff_d: float = 0.0,
    min_modifier: float = 0.50,
    max_modifier: float = 1.20,
) -> pd.DataFrame:
    """Evaluate part-load-ratio COP correction as a publication diagnostic."""
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    df = daily_df.copy()
    q_cool_des = pd.to_numeric(df.get("Q_cool_des_kw", 0.0), errors="coerce").replace(0, np.nan)
    q_heat_des = pd.to_numeric(df.get("Q_heat_des_kw", 0.0), errors="coerce").replace(0, np.nan)
    q = pd.to_numeric(df.get("Q_HVAC_kw", 0.0), errors="coerce").fillna(0.0)
    mode = df.get("mode", pd.Series("cooling", index=df.index)).astype(str)
    design = np.where(mode.eq("cooling"), q_cool_des, q_heat_des)
    plr = (q / np.maximum(design, 1e-9)).clip(lower=0.0, upper=1.5)
    if curve_type == "linear":
        modifier = coeff_a + coeff_b * plr
    elif curve_type == "cubic":
        modifier = coeff_a + coeff_b * plr + coeff_c * plr**2 + coeff_d * plr**3
    else:
        modifier = coeff_a + coeff_b * plr + coeff_c * plr**2
    modifier = np.clip(modifier, float(min_modifier), float(max_modifier))
    cop_base = pd.to_numeric(df.get("COP_eff", 1.0), errors="coerce").fillna(1.0)
    cop_plr = np.maximum(cop_base * modifier, 0.5)
    duration = _duration_col_value(df)
    thermal_kwh_plr = q / np.maximum(cop_plr, 1e-9) * duration
    base_thermal = pd.to_numeric(df.get("thermal_hvac_kwh_period", df.get("energy_kwh_period", df.get("energy_kwh_day", 0.0))), errors="coerce").fillna(0.0)
    out = df[[c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year", "day", "day_of_year", "hour_of_day", "mode"] if c in df.columns]].copy()
    out["Q_HVAC_kw"] = q
    out["design_load_kw"] = design
    out["PLR"] = plr
    out["COP_base"] = cop_base
    out["PLR_modifier"] = modifier
    out["COP_with_PLR"] = cop_plr
    out["thermal_kwh_base"] = base_thermal
    out["thermal_kwh_with_PLR"] = thermal_kwh_plr
    out["thermal_energy_delta_kwh"] = thermal_kwh_plr - base_thermal
    return out


def build_latent_load_analysis(
    daily_df: pd.DataFrame,
    bldg=None,
    cfg=None,
    indoor_rh_pct: float = 50.0,
    indoor_temp_c: float | None = None,
    ventilation_fraction: float = 1.0,
    include_infiltration: bool = True,
    floor_to_floor_m: float = 3.2,
    pressure_kpa: float = 101.325,
) -> pd.DataFrame:
    """Estimate latent cooling load using humidity-ratio difference."""
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()
    df = daily_df.copy()
    area = float(getattr(bldg, "conditioned_area_m2", df.get("area_m2", pd.Series([5000.0])).iloc[0] if "area_m2" in df else 5000.0))
    airflow_m3h_m2 = float(getattr(bldg, "airflow_m3h_m2", 4.0))
    infil_ach = float(getattr(bldg, "infiltration_ach", 0.5))
    rho_air = 1.20
    h_fg = 2501.0  # kJ/kg water
    t_amb = pd.to_numeric(df.get("T_amb_C", 25.0), errors="coerce").fillna(25.0)
    rh = pd.to_numeric(df.get("RH_mean_pct", 60.0), errors="coerce").fillna(60.0)
    occ = pd.to_numeric(df.get("occ", 1.0), errors="coerce").fillna(1.0)
    af = pd.to_numeric(df.get("alpha_flow", 1.0), errors="coerce").fillna(1.0)
    t_indoor = float(indoor_temp_c) if indoor_temp_c is not None else float(getattr(cfg, "T_SET", 23.0) if cfg is not None else 23.0)
    w_out = [humidity_ratio_kgkg(t, r, pressure_kpa) for t, r in zip(t_amb, rh)]
    w_in = humidity_ratio_kgkg(t_indoor, indoor_rh_pct, pressure_kpa)
    vent_m3h = area * airflow_m3h_m2 * af * occ * float(ventilation_fraction)
    infil_m3h = area * float(floor_to_floor_m) * infil_ach if include_infiltration else 0.0
    m_air = (vent_m3h + infil_m3h) / 3600.0 * rho_air
    latent_kw = m_air * h_fg * np.maximum(np.array(w_out) - w_in, 0.0)
    q_cool = pd.to_numeric(df.get("Q_cool_kw", 0.0), errors="coerce").fillna(0.0)
    duration = _duration_col_value(df)
    cop = pd.to_numeric(df.get("COP_eff", 1.0), errors="coerce").fillna(1.0).clip(lower=0.5)
    latent_kwh_thermal = latent_kw * duration
    latent_kwh_electric = latent_kwh_thermal / cop
    out = df[[c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year", "day", "day_of_year", "hour_of_day", "mode"] if c in df.columns]].copy()
    out["T_amb_C"] = t_amb
    out["RH_mean_pct"] = rh
    out["indoor_temp_C"] = t_indoor
    out["indoor_RH_pct"] = float(indoor_rh_pct)
    out["humidity_ratio_outdoor_kgkg"] = w_out
    out["humidity_ratio_indoor_kgkg"] = w_in
    out["ventilation_m3h"] = vent_m3h
    out["infiltration_m3h"] = infil_m3h
    out["latent_cooling_kw"] = latent_kw
    out["sensible_cooling_kw_model"] = q_cool
    out["total_cooling_with_latent_kw"] = q_cool + latent_kw
    out["latent_thermal_kwh_period"] = latent_kwh_thermal
    out["latent_electric_kwh_period"] = latent_kwh_electric
    return out


def build_native_zone_load_table(daily_df: pd.DataFrame, zone_df: Optional[pd.DataFrame], bldg=None) -> pd.DataFrame:
    """Create a stronger zone-level load table by using zone area, density, and schedules.

    This is still reduced-order but more defensible than a pure area split.
    """
    if daily_df is None or daily_df.empty or zone_df is None or len(zone_df) == 0:
        return pd.DataFrame()
    z = zone_df.copy()
    for col in ["zone_name", "zone_type", "area_m2", "occ_density"]:
        if col not in z.columns:
            return pd.DataFrame()
    for c in ["area_m2", "occ_density", "term_factor", "break_factor", "summer_factor"]:
        if c in z.columns:
            z[c] = pd.to_numeric(z[c], errors="coerce")
    area_total = max(float(z["area_m2"].sum()), 1e-9)
    rows = []
    for _, r in z.iterrows():
        area_w = float(r["area_m2"]) / area_total
        occ_w = float(r["area_m2"] * max(r["occ_density"], 0.001))
        ztype = str(r.get("zone_type", "Custom"))
        zname = str(r.get("zone_name", "Zone"))
        local = daily_df.copy()
        # zone factor combines area and occupant-intensity relative to average.
        mean_occ_density = max(float(z["occ_density"].mean()), 1e-9)
        intensity_factor = 0.65 + 0.35 * float(r["occ_density"]) / mean_occ_density
        weight = area_w * intensity_factor
        local["zone_name"] = zname
        local["zone_type"] = ztype
        local["zone_area_m2"] = float(r["area_m2"])
        local["zone_occ_density"] = float(r["occ_density"])
        for src, dst in [("Q_cool_kw", "zone_Q_cool_kw"), ("Q_heat_kw", "zone_Q_heat_kw"), ("Q_HVAC_kw", "zone_Q_HVAC_kw")]:
            local[dst] = pd.to_numeric(local.get(src, 0.0), errors="coerce").fillna(0.0) * weight
        e_col = _period_energy_col(local)
        local["zone_energy_kwh_period"] = pd.to_numeric(local.get(e_col, 0.0), errors="coerce").fillna(0.0) * weight
        if "co2_kg_period" in local.columns:
            local["zone_co2_kg_period"] = pd.to_numeric(local["co2_kg_period"], errors="coerce").fillna(0.0) * weight
        elif "co2_kg_day" in local.columns:
            local["zone_co2_kg_period"] = pd.to_numeric(local["co2_kg_day"], errors="coerce").fillna(0.0) * weight
        else:
            local["zone_co2_kg_period"] = 0.0
        local["zone_comfort_dev_C"] = pd.to_numeric(local.get("comfort_dev_C", 0.0), errors="coerce").fillna(0.0) * (0.90 + 0.20 * intensity_factor)
        keep = [c for c in ["scenario_combo_3axis", "strategy", "severity", "climate", "year", "day", "day_of_year", "hour_of_day", "zone_name", "zone_type", "zone_area_m2", "zone_occ_density", "zone_Q_cool_kw", "zone_Q_heat_kw", "zone_Q_HVAC_kw", "zone_energy_kwh_period", "zone_co2_kg_period", "zone_comfort_dev_C"] if c in local.columns]
        rows.append(local[keep])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_formal_validation_metrics(model_df: pd.DataFrame, reference_df: pd.DataFrame, model_col: str, reference_col: str) -> pd.DataFrame:
    """Return formal validation metrics for model-vs-reference series."""
    if model_df is None or reference_df is None or model_df.empty or reference_df.empty:
        return pd.DataFrame()
    m = pd.to_numeric(model_df[model_col], errors="coerce") if model_col in model_df.columns else pd.Series(dtype=float)
    r = pd.to_numeric(reference_df[reference_col], errors="coerce") if reference_col in reference_df.columns else pd.Series(dtype=float)
    n = int(min(len(m), len(r)))
    if n == 0:
        return pd.DataFrame()
    m = m.iloc[:n].to_numpy(dtype=float)
    r = r.iloc[:n].to_numpy(dtype=float)
    mask = np.isfinite(m) & np.isfinite(r)
    m, r = m[mask], r[mask]
    if len(m) == 0:
        return pd.DataFrame()
    err = m - r
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mbe = float(np.mean(err))
    mean_ref = float(np.mean(r))
    nmbe = float(100.0 * mbe / max(abs(mean_ref), 1e-9))
    cvrmse = float(100.0 * rmse / max(abs(mean_ref), 1e-9))
    mape = float(100.0 * np.mean(np.abs(err / np.maximum(np.abs(r), 1e-9))))
    ss_res = float(np.sum((r - m)**2))
    ss_tot = float(np.sum((r - np.mean(r))**2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-9))
    return pd.DataFrame([{
        "model_column": model_col,
        "reference_column": reference_col,
        "n": int(len(m)),
        "MBE": mbe,
        "NMBE_pct": nmbe,
        "RMSE": rmse,
        "CVRMSE_pct": cvrmse,
        "MAE": mae,
        "MAPE_pct": mape,
        "R2": r2,
    }])


def build_global_sensitivity_from_samples(samples_df: pd.DataFrame) -> pd.DataFrame:
    """Approximate global sensitivity using absolute Pearson/Spearman correlations.

    It is intended as a lightweight global screening method over robustness samples.
    """
    if samples_df is None or samples_df.empty:
        return pd.DataFrame()
    input_cols = [c for c in samples_df.columns if c.startswith("input_")]
    kpi_cols = [c for c in ["Total Energy MWh", "Total Cost USD", "Total CO2 tonne", "Mean COP", "Mean Degradation Index", "Mean Comfort Deviation C"] if c in samples_df.columns]
    rows = []
    for inp in input_cols:
        x = pd.to_numeric(samples_df[inp], errors="coerce")
        for kpi in kpi_cols:
            y = pd.to_numeric(samples_df[kpi], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                continue
            pear = float(x[valid].corr(y[valid], method="pearson"))
            spear = float(x[valid].corr(y[valid], method="spearman"))
            rows.append({"input_parameter": inp.replace("input_", ""), "kpi": kpi, "pearson_r": pear, "spearman_r": spear, "abs_pearson": abs(pear), "abs_spearman": abs(spear), "importance": float(np.nanmean([abs(pear), abs(spear)]))})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("importance", ascending=False)
    return out

# -----------------------------
# EMS schedules and multi-objective optimization utilities
# -----------------------------
def build_operation_schedule_template() -> pd.DataFrame:
    """Default editable operation schedule used by the Streamlit Schedule tab."""
    return pd.DataFrame([
        {"day_type": "Weekday", "start_hour": 0.0, "end_hour": 6.0, "occupied": 0, "occ_multiplier": 0.10, "setpoint_shift_C": 2.0, "airflow_factor": 0.55, "cooling_allowed": 0, "heating_allowed": 1, "demand_response": 0},
        {"day_type": "Weekday", "start_hour": 6.0, "end_hour": 8.0, "occupied": 1, "occ_multiplier": 0.50, "setpoint_shift_C": -0.5, "airflow_factor": 0.85, "cooling_allowed": 1, "heating_allowed": 1, "demand_response": 0},
        {"day_type": "Weekday", "start_hour": 8.0, "end_hour": 16.0, "occupied": 1, "occ_multiplier": 1.00, "setpoint_shift_C": 0.0, "airflow_factor": 1.00, "cooling_allowed": 1, "heating_allowed": 1, "demand_response": 0},
        {"day_type": "Weekday", "start_hour": 16.0, "end_hour": 19.0, "occupied": 1, "occ_multiplier": 0.55, "setpoint_shift_C": 0.5, "airflow_factor": 0.80, "cooling_allowed": 1, "heating_allowed": 1, "demand_response": 0},
        {"day_type": "Weekday", "start_hour": 19.0, "end_hour": 24.0, "occupied": 0, "occ_multiplier": 0.15, "setpoint_shift_C": 2.0, "airflow_factor": 0.55, "cooling_allowed": 0, "heating_allowed": 1, "demand_response": 0},
        {"day_type": "Weekend", "start_hour": 0.0, "end_hour": 24.0, "occupied": 0, "occ_multiplier": 0.20, "setpoint_shift_C": 2.5, "airflow_factor": 0.55, "cooling_allowed": 0, "heating_allowed": 1, "demand_response": 0},
    ])


def validate_operation_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate schedule rows for use by the engine."""
    if schedule_df is None or schedule_df.empty:
        return build_operation_schedule_template()
    df = schedule_df.copy()
    required = {
        "day_type": "All",
        "start_hour": 0.0,
        "end_hour": 24.0,
        "occupied": 1,
        "occ_multiplier": 1.0,
        "setpoint_shift_C": 0.0,
        "airflow_factor": 1.0,
        "cooling_allowed": 1,
        "heating_allowed": 1,
        "demand_response": 0,
    }
    for col, default in required.items():
        if col not in df.columns:
            df[col] = default
    df["day_type"] = df["day_type"].astype(str).replace({"nan": "All"})
    for col in ["start_hour", "end_hour", "occ_multiplier", "setpoint_shift_C", "airflow_factor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(required[col])
    df["start_hour"] = df["start_hour"].clip(0, 24)
    df["end_hour"] = df["end_hour"].clip(0, 24)
    df["occ_multiplier"] = df["occ_multiplier"].clip(0, 2.0)
    df["airflow_factor"] = df["airflow_factor"].clip(0.1, 1.5)
    for col in ["occupied", "cooling_allowed", "heating_allowed", "demand_response"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(required[col]).astype(int).clip(0, 1)
    return df[list(required.keys())]


def _nondominated_mask(df: pd.DataFrame, objective_cols: list[str], minimize_cols: Optional[list[str]] = None) -> np.ndarray:
    if df.empty or not objective_cols:
        return np.array([], dtype=bool)
    minimize_cols = minimize_cols or objective_cols
    vals = df[objective_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # Convert maximize objectives to minimize by negating. Current objective list is all minimization.
    n = len(vals)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                mask[i] = False
                break
    return mask


def run_multi_objective_search(
    output_dir: str | Path,
    bldg,
    cfg,
    weather_mode: str = "synthetic",
    epw_path: str | None = None,
    csv_path: str | None = None,
    weather_df: Optional[pd.DataFrame] = None,
    operation_schedule_df: Optional[pd.DataFrame] = None,
    fixed_strategy: str = "S3",
    fixed_severity: str = "Moderate",
    fixed_climate: str = "C0_Baseline",
    optimizer_name: str = "Weighted random search",
    n_candidates: int = 12,
    analysis_years: int = 1,
    random_state: int = 42,
    weight_energy: float = 0.35,
    weight_degradation: float = 0.25,
    weight_comfort: float = 0.25,
    weight_carbon: float = 0.15,
) -> Dict[str, str]:
    """Run a lightweight multi-objective search over EMS/control parameters.

    This is designed for publication screening and reproducible comparisons, not as a
    heavy external optimizer. Users can label the optimizer they want to emulate or
    compare, while the output stores all decision variables and objective values.
    """
    from dataclasses import asdict
    from hvac_v3_engine import HVACConfig, run_scenario_model

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(random_state))
    base = HVACConfig(**asdict(cfg))
    base.years = int(max(1, analysis_years))
    weights = np.array([weight_energy, weight_degradation, weight_comfort, weight_carbon], dtype=float)
    weights = weights / max(float(weights.sum()), 1e-9)

    candidates = []
    opt_l = str(optimizer_name).lower()
    if "grid" in opt_l:
        t_vals = np.linspace(max(base.T_SP_MIN, base.T_SET - 1.5), min(base.T_SP_MAX, base.T_SET + 2.0), max(2, int(np.ceil(n_candidates ** 0.5))))
        af_vals = np.linspace(max(base.AF_MIN, 0.45), min(base.AF_MAX, 1.15), max(2, int(np.ceil(n_candidates ** 0.5))))
        for t in t_vals:
            for af in af_vals:
                candidates.append({"T_SET": float(t), "AF_MIN": float(min(base.AF_MIN, af)), "AF_MAX": float(max(base.AF_MIN, af)), "EMS_DR_SETPOINT_SHIFT_C": 1.0, "EMS_LOW_OCC_AIRFLOW_FACTOR": float(af)})
                if len(candidates) >= n_candidates:
                    break
            if len(candidates) >= n_candidates:
                break
    else:
        for _ in range(int(n_candidates)):
            candidates.append({
                "T_SET": float(rng.uniform(max(base.T_SP_MIN, base.T_SET - 2.0), min(base.T_SP_MAX, base.T_SET + 2.5))),
                "AF_MIN": float(rng.uniform(0.45, 0.75)),
                "AF_MAX": float(rng.uniform(0.85, 1.15)),
                "EMS_DR_SETPOINT_SHIFT_C": float(rng.uniform(0.5, 2.5)),
                "EMS_LOW_OCC_AIRFLOW_FACTOR": float(rng.uniform(0.45, 0.85)),
                "EMS_NIGHT_SETPOINT_SHIFT_C": float(rng.uniform(1.0, 3.0)),
                "EMS_ECONOMIZER_COOLING_REDUCTION": float(rng.uniform(0.05, 0.35)),
            })
    # Always include the base candidate for comparison.
    candidates.insert(0, {"T_SET": float(base.T_SET), "AF_MIN": float(base.AF_MIN), "AF_MAX": float(base.AF_MAX), "EMS_DR_SETPOINT_SHIFT_C": float(base.EMS_DR_SETPOINT_SHIFT_C), "EMS_LOW_OCC_AIRFLOW_FACTOR": float(base.EMS_LOW_OCC_AIRFLOW_FACTOR), "EMS_NIGHT_SETPOINT_SHIFT_C": float(base.EMS_NIGHT_SETPOINT_SHIFT_C), "EMS_ECONOMIZER_COOLING_REDUCTION": float(base.EMS_ECONOMIZER_COOLING_REDUCTION)})

    rows = []
    for i, cand in enumerate(candidates):
        ccfg = HVACConfig(**asdict(base))
        for k, v in cand.items():
            if hasattr(ccfg, k):
                setattr(ccfg, k, v)
        if ccfg.AF_MAX < ccfg.AF_MIN:
            ccfg.AF_MAX = ccfg.AF_MIN
        run_dir = out / f"candidate_{i:03d}"
        result = run_scenario_model(
            output_dir=run_dir,
            axis_mode="baseline_scenario",
            bldg=bldg,
            cfg=ccfg,
            weather_mode=weather_mode,
            epw_path=epw_path,
            csv_path=csv_path,
            weather_df=weather_df,
            fixed_strategy=fixed_strategy,
            fixed_severity=fixed_severity,
            fixed_climate=fixed_climate,
            random_state=random_state + i,
            include_baseline_layer=False,
            include_baseline_as_scenario=False,
            degradation_model=getattr(ccfg, "degradation_model", "physics"),
            time_step_hours=getattr(ccfg, "TIME_STEP_HOURS", 24.0),
            operation_schedule_df=operation_schedule_df,
        )
        summ = pd.read_csv(result["summary_csv"])
        r = summ.iloc[0].to_dict() if not summ.empty else {}
        row = {"candidate_id": i, "optimizer_name": optimizer_name, **cand}
        row.update({k: r.get(k, np.nan) for k in ["Total Energy MWh", "Total Cost USD", "Total CO2 tonne", "Mean COP", "Mean Degradation Index", "Mean Comfort Deviation C", "Occupied Discomfort Days"]})
        rows.append(row)

    df = pd.DataFrame(rows)
    objectives = ["Total Energy MWh", "Mean Degradation Index", "Mean Comfort Deviation C", "Total CO2 tonne"]
    for col in objectives:
        if col in df.columns:
            c = pd.to_numeric(df[col], errors="coerce")
            df[f"norm_{col}"] = (c - c.min()) / max(float(c.max() - c.min()), 1e-9)
    norm_cols = [f"norm_{c}" for c in objectives if f"norm_{c}" in df.columns]
    if len(norm_cols) == 4:
        df["weighted_objective"] = df[norm_cols].to_numpy(dtype=float).dot(weights)
    else:
        df["weighted_objective"] = np.nan
    df["pareto_candidate"] = _nondominated_mask(df, objectives).astype(int) if all(c in df.columns for c in objectives) else 0
    df = df.sort_values(["pareto_candidate", "weighted_objective"], ascending=[False, True])
    results_csv = out / "multi_objective_candidates.csv"
    pareto_csv = out / "multi_objective_pareto.csv"
    metadata_json = out / "multi_objective_metadata.json"
    df.to_csv(results_csv, index=False)
    df[df["pareto_candidate"] == 1].to_csv(pareto_csv, index=False)
    metadata = {
        "optimizer_name": optimizer_name,
        "n_candidates_requested": int(n_candidates),
        "n_candidates_evaluated": int(len(df)),
        "fixed_strategy": fixed_strategy,
        "fixed_severity": fixed_severity,
        "fixed_climate": fixed_climate,
        "weights": {"energy": float(weights[0]), "degradation": float(weights[1]), "comfort": float(weights[2]), "carbon": float(weights[3])},
        "analysis_years": int(analysis_years),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"candidates_csv": str(results_csv), "pareto_csv": str(pareto_csv), "metadata_json": str(metadata_json), "output_dir": str(out)}


# -----------------------------
# Advanced HVAC Control Library
# -----------------------------
def build_advanced_control_candidates(base_setup: dict | None = None) -> pd.DataFrame:
    """Return a transparent control-library table for advanced HVAC control strategies.

    These are deployable parameter recommendations that can be applied to the EMS/schedule layer.
    MPC and RL rows are intentionally marked experimental because they require forecasts/training data
    and should not be represented as fully validated live controllers.
    """
    rows = [
        {
            "control_id": "G36_STYLE",
            "control_name": "ASHRAE G36-style high-performance sequence",
            "status": "implemented_rule_based_template",
            "ems_mode": "Smart hybrid",
            "use_occ_reset": True,
            "use_night_setback": True,
            "use_demand_response": False,
            "use_economizer": True,
            "use_optimum_start": True,
            "setpoint_shift_C": 0.8,
            "airflow_factor": 0.75,
            "economizer_reduction": 0.25,
            "maintenance_bias": "normal",
            "description": "Combines occupancy reset, night setback, economizer/free-cooling, and optimum start as a reduced-order analogue of high-performance AHU/VAV control sequences.",
        },
        {
            "control_id": "DCV",
            "control_name": "Demand-controlled ventilation",
            "status": "implemented_rule_based_template",
            "ems_mode": "Occupancy-based",
            "use_occ_reset": True,
            "use_night_setback": False,
            "use_demand_response": False,
            "use_economizer": False,
            "use_optimum_start": False,
            "setpoint_shift_C": 0.5,
            "airflow_factor": 0.60,
            "economizer_reduction": 0.00,
            "maintenance_bias": "normal",
            "description": "Reduces airflow during low occupancy and raises ventilation/airflow only when occupancy or CO2 requires it. In this reduced-order version it is represented through occupancy-based airflow reset.",
        },
        {
            "control_id": "FAULT_ADAPTIVE",
            "control_name": "Fault-adaptive control",
            "status": "implemented_recommendation_template",
            "ems_mode": "Smart hybrid",
            "use_occ_reset": True,
            "use_night_setback": True,
            "use_demand_response": False,
            "use_economizer": True,
            "use_optimum_start": False,
            "setpoint_shift_C": 0.6,
            "airflow_factor": 0.80,
            "economizer_reduction": 0.15,
            "maintenance_bias": "aggressive_when_faulted",
            "description": "Uses fault indicators such as high pressure drop, low COP, high fan power, or comfort faults to recommend maintenance and safer control settings.",
        },
        {
            "control_id": "DEGRADATION_MAINT_OPT",
            "control_name": "Degradation-aware maintenance optimizer",
            "status": "implemented_recommendation_template",
            "ems_mode": "Smart hybrid",
            "use_occ_reset": True,
            "use_night_setback": False,
            "use_demand_response": False,
            "use_economizer": True,
            "use_optimum_start": False,
            "setpoint_shift_C": 0.4,
            "airflow_factor": 0.85,
            "economizer_reduction": 0.10,
            "maintenance_bias": "energy_cost_trigger",
            "description": "Compares expected energy penalty from degradation with maintenance cost and recommends filter replacement or HX cleaning before penalties become excessive.",
        },
        {
            "control_id": "CARBON_PRICE_AWARE",
            "control_name": "Carbon-aware / price-aware control",
            "status": "implemented_rule_based_template",
            "ems_mode": "Demand response",
            "use_occ_reset": True,
            "use_night_setback": True,
            "use_demand_response": True,
            "use_economizer": True,
            "use_optimum_start": True,
            "setpoint_shift_C": 1.4,
            "airflow_factor": 0.70,
            "economizer_reduction": 0.20,
            "maintenance_bias": "normal",
            "description": "Shifts flexible HVAC operation away from high-price or high-carbon hours using pre-cooling and demand-response setpoint reset.",
        },
        {
            "control_id": "PEAK_LIMITER",
            "control_name": "Peak demand limiter",
            "status": "implemented_rule_based_template",
            "ems_mode": "Demand response",
            "use_occ_reset": True,
            "use_night_setback": False,
            "use_demand_response": True,
            "use_economizer": True,
            "use_optimum_start": True,
            "setpoint_shift_C": 1.7,
            "airflow_factor": 0.65,
            "economizer_reduction": 0.15,
            "maintenance_bias": "normal",
            "description": "Limits peak HVAC demand by setpoint reset, airflow reduction, and pre-cooling before peak periods.",
        },
        {
            "control_id": "ZONE_PRIORITY",
            "control_name": "Zone priority control",
            "status": "implemented_allocation_template",
            "ems_mode": "Custom scheduled",
            "use_occ_reset": True,
            "use_night_setback": True,
            "use_demand_response": False,
            "use_economizer": True,
            "use_optimum_start": False,
            "setpoint_shift_C": 0.5,
            "airflow_factor": 0.80,
            "economizer_reduction": 0.10,
            "maintenance_bias": "normal",
            "description": "Allocates stricter comfort and airflow priority to lecture halls/labs and relaxed control to corridors/service zones through schedule and zone-level outputs.",
        },
        {
            "control_id": "HYBRID_CONTROLLER",
            "control_name": "Hybrid degradation-aware EMS controller",
            "status": "implemented_rule_based_template",
            "ems_mode": "Smart hybrid",
            "use_occ_reset": True,
            "use_night_setback": True,
            "use_demand_response": True,
            "use_economizer": True,
            "use_optimum_start": True,
            "setpoint_shift_C": 1.0,
            "airflow_factor": 0.72,
            "economizer_reduction": 0.25,
            "maintenance_bias": "aggressive_when_degraded",
            "description": "Combines schedule, occupancy reset, demand response, economizer, degradation-aware maintenance, and safety comfort limits.",
        },
        {
            "control_id": "MPC_EXPERIMENTAL",
            "control_name": "Model Predictive Control experimental template",
            "status": "experimental_not_live_control",
            "ems_mode": "Smart hybrid",
            "use_occ_reset": True,
            "use_night_setback": True,
            "use_demand_response": True,
            "use_economizer": True,
            "use_optimum_start": True,
            "setpoint_shift_C": 1.0,
            "airflow_factor": 0.75,
            "economizer_reduction": 0.20,
            "maintenance_bias": "forecast_based_placeholder",
            "description": "Experimental design for future receding-horizon optimization. Requires weather/occupancy/price forecasts and solver integration before it should be used as an autonomous controller.",
        },
        {
            "control_id": "RL_EXPERIMENTAL",
            "control_name": "Reinforcement Learning dataset/export template",
            "status": "experimental_dataset_only",
            "ems_mode": "Smart hybrid",
            "use_occ_reset": True,
            "use_night_setback": True,
            "use_demand_response": True,
            "use_economizer": True,
            "use_optimum_start": False,
            "setpoint_shift_C": 0.8,
            "airflow_factor": 0.75,
            "economizer_reduction": 0.15,
            "maintenance_bias": "learned_policy_placeholder",
            "description": "Prepares state/action/reward design for offline RL. It does not train or deploy a validated RL controller in the current software.",
        },
    ]
    return pd.DataFrame(rows)


def build_control_objective_table(candidates: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Score control-library candidates using transparent engineering assumptions.

    This is a lightweight pre-screening score, not a replacement for Scenario Modeling or
    Multi-Objective Optimization. Lower score is better.
    """
    if weights is None:
        weights = {"energy": 0.35, "comfort": 0.25, "degradation": 0.20, "carbon": 0.10, "fault_risk": 0.10}
    df = candidates.copy()
    # Qualitative-to-quantitative pre-screening estimates.
    df["estimated_energy_saving_pct"] = (
        5.0 * df["use_occ_reset"].astype(float)
        + 4.0 * df["use_night_setback"].astype(float)
        + 4.5 * df["use_economizer"].astype(float)
        + 3.0 * df["use_demand_response"].astype(float)
        + 2.0 * df["use_optimum_start"].astype(float)
        + 3.0 * (1.0 - df["airflow_factor"].astype(float)).clip(0, 1)
    ).clip(0, 35)
    df["estimated_comfort_risk_score"] = (
        1.0
        + 1.2 * df["use_demand_response"].astype(float)
        + 0.7 * df["use_night_setback"].astype(float)
        + 0.9 * df["setpoint_shift_C"].astype(float).abs()
        - 0.6 * df["use_optimum_start"].astype(float)
    ).clip(0, 10)
    df["estimated_degradation_risk_score"] = (
        3.0
        + 1.5 * (1.0 - df["airflow_factor"].astype(float)).clip(0, 1)
        - 0.9 * df["use_economizer"].astype(float)
        - 0.8 * df["maintenance_bias"].astype(str).str.contains("aggressive|energy_cost", case=False, regex=True).astype(float)
    ).clip(0, 10)
    df["estimated_carbon_saving_pct"] = df["estimated_energy_saving_pct"] * (1.0 + 0.15 * df["use_demand_response"].astype(float))
    df["fault_adaptiveness_score"] = (
        2.0
        + 4.0 * df["control_id"].astype(str).str.contains("FAULT|HYBRID", regex=True).astype(float)
        + 2.0 * df["maintenance_bias"].astype(str).str.contains("aggressive|fault|degraded", case=False, regex=True).astype(float)
    ).clip(0, 10)
    # Normalize for score where lower is better.
    e = 1.0 - df["estimated_energy_saving_pct"] / max(df["estimated_energy_saving_pct"].max(), 1e-9)
    c = df["estimated_comfort_risk_score"] / max(df["estimated_comfort_risk_score"].max(), 1e-9)
    d = df["estimated_degradation_risk_score"] / max(df["estimated_degradation_risk_score"].max(), 1e-9)
    co = 1.0 - df["estimated_carbon_saving_pct"] / max(df["estimated_carbon_saving_pct"].max(), 1e-9)
    f = 1.0 - df["fault_adaptiveness_score"] / max(df["fault_adaptiveness_score"].max(), 1e-9)
    df["weighted_control_score"] = (
        weights.get("energy", 0.35) * e
        + weights.get("comfort", 0.25) * c
        + weights.get("degradation", 0.20) * d
        + weights.get("carbon", 0.10) * co
        + weights.get("fault_risk", 0.10) * f
    )
    return df.sort_values("weighted_control_score").reset_index(drop=True)


def build_mpc_experimental_template(horizon_hours: int = 24, control_step_hours: int = 1) -> pd.DataFrame:
    hours = list(range(0, int(horizon_hours), int(max(control_step_hours, 1))))
    return pd.DataFrame({
        "horizon_step": range(len(hours)),
        "hour_ahead": hours,
        "forecast_T_amb_C": np.nan,
        "forecast_RH_pct": np.nan,
        "forecast_GHI_Wm2": np.nan,
        "forecast_occupancy_factor": np.nan,
        "forecast_price_usd_kwh": np.nan,
        "forecast_grid_co2_kg_kwh": np.nan,
        "decision_cooling_setpoint_C": np.nan,
        "decision_airflow_factor": np.nan,
        "decision_economizer_enabled": np.nan,
        "decision_dr_enabled": np.nan,
    })


def build_rl_experimental_dataset_spec() -> pd.DataFrame:
    rows = [
        {"field": "state_T_amb_C", "type": "state", "description": "Outdoor dry-bulb temperature"},
        {"field": "state_RH_pct", "type": "state", "description": "Outdoor/zone humidity signal"},
        {"field": "state_GHI_Wm2", "type": "state", "description": "Solar radiation"},
        {"field": "state_occ", "type": "state", "description": "Occupancy factor or count"},
        {"field": "state_hour", "type": "state", "description": "Hour of day"},
        {"field": "state_delta", "type": "state", "description": "HVAC degradation index"},
        {"field": "state_COP", "type": "state", "description": "Current effective COP"},
        {"field": "action_setpoint_shift_C", "type": "action", "description": "Control action: setpoint reset"},
        {"field": "action_airflow_factor", "type": "action", "description": "Control action: airflow factor"},
        {"field": "action_economizer", "type": "action", "description": "Control action: economizer enable/disable"},
        {"field": "action_maintenance", "type": "action", "description": "Control action: maintenance trigger"},
        {"field": "reward", "type": "reward", "description": "Negative weighted sum of energy cost, comfort violation, degradation, CO2, and fault risk"},
    ]
    return pd.DataFrame(rows)
