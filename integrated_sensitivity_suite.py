from __future__ import annotations

"""Comprehensive local/global sensitivity studio for the official HVAC v3 engine.

The module is deliberately independent of ChillerPlant.csv. Every model evaluation
calls the same ``simulate_combo`` function used by the main application.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
import gc
import hashlib
import io
import json
import os
import math
import re
import shutil
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import streamlit as st
except Exception:  # Allows headless engine tests before UI dependencies are installed.
    st = None

from hvac_v3_engine import (
    BuildingSpec,
    HVACConfig,
    SCENARIOS,
    SEVERITY_LEVELS,
    CLIMATE_LEVELS,
    _load_base_weather,
    aggregate_zone_occupancy,
    simulate_combo,
)


STANDARD_KPIS = [
    "total_energy_MWh",
    "thermal_hvac_MWh",
    "fan_MWh",
    "pump_MWh",
    "auxiliary_MWh",
    "total_cost_USD",
    "co2_tonne",
    "mean_COP",
    "mean_residual_delta",
    "mean_comfort_dev_C",
    "occupied_discomfort_days",
    "filter_replacements",
    "hx_cleanings",
]

KPI_LABELS = {
    "total_energy_MWh": "Total HVAC-system energy (MWh)",
    "thermal_hvac_MWh": "Thermal HVAC electricity (MWh)",
    "fan_MWh": "Fan energy (MWh)",
    "pump_MWh": "Pump energy (MWh)",
    "auxiliary_MWh": "Auxiliary energy (MWh)",
    "total_cost_USD": "Total operating cost (USD)",
    "co2_tonne": "Operational emissions (tCO₂)",
    "mean_COP": "Mean effective COP",
    "mean_residual_delta": "Mean residual degradation index",
    "mean_comfort_dev_C": "Mean comfort deviation (°C)",
    "occupied_discomfort_days": "Occupied discomfort days",
    "filter_replacements": "Filter replacements",
    "hx_cleanings": "Heat-exchanger cleanings",
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    group: str
    target: str  # building | config
    lower: float
    baseline: float
    upper: float
    integer: bool = False
    enabled: bool = True
    description: str = ""

    def validate(self) -> None:
        vals = np.asarray([self.lower, self.baseline, self.upper], dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Non-finite bounds for {self.name}.")
        if self.lower >= self.upper:
            raise ValueError(f"Lower bound must be smaller than upper bound for {self.name}.")
        if not self.lower <= self.baseline <= self.upper:
            raise ValueError(f"Baseline must lie inside bounds for {self.name}.")
        if self.target not in {"building", "config"}:
            raise ValueError(f"Invalid target for {self.name}: {self.target}")


def _bounded_triplet(base: float, low_factor: float, high_factor: float, lo: float | None = None, hi: float | None = None) -> tuple[float, float, float]:
    lower = base * low_factor
    upper = base * high_factor
    if abs(base) < 1e-15:
        lower, upper = 0.0, max(high_factor - 1.0, 0.10)
    if lo is not None:
        lower = max(lower, lo)
        base = max(base, lo)
    if hi is not None:
        upper = min(upper, hi)
        base = min(base, hi)
    if lower >= upper:
        upper = lower + max(abs(base) * 0.1, 1e-6)
    return float(lower), float(base), float(upper)


def default_parameter_specs(bldg: BuildingSpec, cfg: HVACConfig) -> list[ParameterSpec]:
    raw: list[tuple[str, str, str, str, tuple[float, float, float], bool, str]] = []

    def add(name: str, label: str, group: str, target: str, triplet: tuple[float, float, float], integer: bool = False, description: str = ""):
        raw.append((name, label, group, target, triplet, integer, description))

    add("occupancy_density_p_m2", "Occupancy density", "Building loads", "building", _bounded_triplet(bldg.occupancy_density_p_m2, 0.80, 1.20, 0.0001), description="Peak occupants per square metre.")
    add("lighting_w_m2", "Lighting power density", "Building loads", "building", _bounded_triplet(bldg.lighting_w_m2, 0.80, 1.20, 0.0), description="Lighting sensible gain and electricity-related load driver.")
    add("equipment_w_m2", "Equipment power density", "Building loads", "building", _bounded_triplet(bldg.equipment_w_m2, 0.80, 1.20, 0.0), description="Plug and laboratory equipment sensible gain.")
    add("airflow_m3h_m2", "Airflow intensity", "Air system", "building", _bounded_triplet(bldg.airflow_m3h_m2, 0.80, 1.20, 0.01), description="Nominal supply airflow per conditioned area.")
    add("infiltration_ach", "Infiltration rate", "Envelope and weather", "building", _bounded_triplet(bldg.infiltration_ach, 0.70, 1.30, 0.0), description="Outdoor-air infiltration in air changes per hour.")
    add("cooling_intensity_w_m2", "Cooling design intensity", "HVAC sizing", "building", _bounded_triplet(bldg.cooling_intensity_w_m2, 0.85, 1.15, 1.0), description="Installed/design cooling capacity per area.")
    add("wall_u", "Wall U-value", "Envelope and weather", "building", _bounded_triplet(bldg.wall_u, 0.80, 1.20, 0.01), description="Opaque-wall thermal transmittance.")
    add("shgc", "Window SHGC", "Envelope and weather", "building", _bounded_triplet(bldg.shgc, 0.80, 1.20, 0.01, 0.95), description="Solar heat-gain coefficient.")

    add("COP_COOL_NOM", "Nominal cooling COP", "HVAC performance", "config", _bounded_triplet(cfg.COP_COOL_NOM, 0.85, 1.15, 0.8), description="Clean nominal cooling coefficient of performance.")
    add("FAN_EFF", "Fan total efficiency", "HVAC performance", "config", _bounded_triplet(cfg.FAN_EFF, 0.90, 1.10, 0.10, 0.95), description="Total fan efficiency.")
    add("COP_AGING_RATE", "COP ageing rate", "Equipment ageing", "config", _bounded_triplet(cfg.COP_AGING_RATE, 0.50, 1.50, 0.0), description="Annual irreversible COP deterioration term.")
    add("RF_STAR", "Fouling asymptote RF*", "Heat-exchanger fouling", "config", _bounded_triplet(cfg.RF_STAR, 0.70, 1.30, 1e-10), description="Asymptotic heat-exchanger fouling resistance.")
    add("B_FOUL", "Fouling growth constant", "Heat-exchanger fouling", "config", _bounded_triplet(cfg.B_FOUL, 0.70, 1.30, 0.0), description="Exponential fouling-growth coefficient.")
    add("DUST_RATE", "Filter dust accumulation rate", "Filter clogging", "config", _bounded_triplet(cfg.DUST_RATE, 0.70, 1.30, 0.0), description="Dust accumulation per model day at nominal airflow.")
    add("K_CLOG", "Filter clogging coefficient", "Filter clogging", "config", _bounded_triplet(cfg.K_CLOG, 0.70, 1.30, 0.0), description="Pressure-drop increase per accumulated dust unit.")
    add("DEG_TRIGGER", "S3 degradation trigger", "Maintenance policy", "config", (max(0.0, cfg.DEG_TRIGGER - 0.10), cfg.DEG_TRIGGER, cfg.DEG_TRIGGER + 0.10), description="Composite degradation threshold for S3 intervention.")
    add("RF_WARN", "Fouling warning threshold", "Maintenance policy", "config", _bounded_triplet(cfg.RF_WARN, 0.80, 1.20, 1e-10), description="Heat-exchanger warning threshold used by S3.")
    add("DP_WARN", "Filter pressure warning", "Maintenance policy", "config", _bounded_triplet(cfg.DP_WARN, 0.85, 1.15, cfg.DP_CLEAN + 1.0), description="Filter pressure-drop warning threshold used by S3.")
    add("FILTER_INTERVAL", "Preventive filter interval", "Maintenance policy", "config", _bounded_triplet(float(cfg.FILTER_INTERVAL), 0.75, 1.25, 1.0), True, "Scheduled filter replacement interval in days.")
    add("HX_INTERVAL", "Preventive HX interval", "Maintenance policy", "config", _bounded_triplet(float(cfg.HX_INTERVAL), 0.75, 1.25, 1.0), True, "Scheduled heat-exchanger cleaning interval in days.")
    add("E_PRICE", "Electricity tariff", "Economics and carbon", "config", _bounded_triplet(cfg.E_PRICE, 0.70, 1.30, 0.0), description="Electricity tariff in USD/kWh.")
    add("CO2_FACTOR", "Grid emission factor", "Economics and carbon", "config", _bounded_triplet(cfg.CO2_FACTOR, 0.75, 1.25, 0.0), description="Grid emissions in kgCO₂/kWh.")
    add("COST_FILTER", "Filter replacement cost", "Economics and carbon", "config", _bounded_triplet(cfg.COST_FILTER, 0.70, 1.30, 0.0), description="Cost per filter replacement.")
    add("COST_HX", "Heat-exchanger cleaning cost", "Economics and carbon", "config", _bounded_triplet(cfg.COST_HX, 0.70, 1.30, 0.0), description="Cost per heat-exchanger cleaning.")
    add("APO_POP", "S3 candidate population", "S3 optimizer", "config", (min(12.0, float(cfg.APO_POP)), float(cfg.APO_POP), max(30.0, float(cfg.APO_POP) + 1.0)), True, "Candidate controls evaluated in each S3 iteration.")
    add("APO_ITERS", "S3 optimizer iterations", "S3 optimizer", "config", (min(5.0, float(cfg.APO_ITERS)), float(cfg.APO_ITERS), max(20.0, float(cfg.APO_ITERS) + 1.0)), True, "Stochastic search iterations per control step.")

    specs = [ParameterSpec(name, label, group, target, *triplet, integer=integer, description=desc) for name, label, group, target, triplet, integer, desc in raw]
    for spec in specs:
        spec.validate()
    return specs


def specs_to_dataframe(specs: Iterable[ParameterSpec]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in specs])[["enabled", "name", "label", "group", "target", "lower", "baseline", "upper", "integer", "description"]]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def specs_from_dataframe(df: pd.DataFrame) -> list[ParameterSpec]:
    required = {"name", "label", "group", "target", "lower", "baseline", "upper"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Parameter table is missing columns: {sorted(missing)}")
    specs: list[ParameterSpec] = []
    for _, row in df.iterrows():
        enabled = _as_bool(row.get("enabled", True), True)
        spec = ParameterSpec(
            name=str(row["name"]).strip(),
            label=str(row["label"]).strip(),
            group=str(row["group"]).strip(),
            target=str(row["target"]).strip().lower(),
            lower=float(row["lower"]),
            baseline=float(row["baseline"]),
            upper=float(row["upper"]),
            integer=_as_bool(row.get("integer", False), False),
            enabled=enabled,
            description=str(row.get("description", "")),
        )
        spec.validate()
        if enabled:
            specs.append(spec)
    if not specs:
        raise ValueError("At least one sensitivity parameter must be enabled.")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("Parameter names must be unique.")
    return specs


def _slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text)).strip("_")
    return value[:90] or "result"


def _clone_bldg(bldg: BuildingSpec) -> BuildingSpec:
    return BuildingSpec(**asdict(bldg))


def _clone_cfg(cfg: HVACConfig) -> HVACConfig:
    return HVACConfig(**asdict(cfg))


def _standardize_summary(summary: dict[str, Any]) -> dict[str, float]:
    mapping = {
        "total_energy_MWh": "Total Energy MWh",
        "thermal_hvac_MWh": "Total Thermal HVAC Energy MWh",
        "fan_MWh": "Total Fan Energy MWh",
        "pump_MWh": "Total Pump Energy MWh",
        "auxiliary_MWh": "Total Auxiliary Energy MWh",
        "total_cost_USD": "Total Cost USD",
        "co2_tonne": "Total CO2 tonne",
        "mean_COP": "Mean COP",
        "mean_residual_delta": "Mean Degradation Index",
        "mean_comfort_dev_C": "Mean Comfort Deviation C",
        "occupied_discomfort_days": "Occupied Discomfort Days",
        "filter_replacements": "Filter Replacements count",
        "hx_cleanings": "HX Cleanings count",
    }
    out: dict[str, float] = {}
    for new, old in mapping.items():
        try:
            val = float(summary.get(old, np.nan))
        except Exception:
            val = np.nan
        out[new] = val
    return out


def _value_from_unit(spec: ParameterSpec, x: float) -> float:
    value = spec.lower + float(x) * (spec.upper - spec.lower)
    if spec.integer:
        value = float(int(round(value)))
    return float(np.clip(value, spec.lower, spec.upper))


def _params_from_unit(specs: list[ParameterSpec], x: np.ndarray) -> dict[str, float]:
    return {spec.name: _value_from_unit(spec, x[i]) for i, spec in enumerate(specs)}


class OfficialHVACRunner:
    """Repeated scalar evaluations through the official HVAC v3 ``simulate_combo`` solver."""

    def __init__(
        self,
        bldg: BuildingSpec,
        cfg: HVACConfig,
        analysis_years: int,
        weather_mode: str,
        epw_path: str | None,
        csv_path: str | None,
        weather_df: pd.DataFrame | None,
        random_state: int,
        zone_df: pd.DataFrame | None,
        operation_schedule_df: pd.DataFrame | None,
        degradation_model: str,
        time_step_hours: float,
        disk_cache_dir: str | Path | None = None,
        summary_only: bool = True,
        max_memory_cache_entries: int = 256,
    ):
        self.base_bldg = _clone_bldg(bldg)
        self.base_cfg = _clone_cfg(cfg)
        self.base_cfg.years = int(max(1, analysis_years))
        self.base_cfg.TIME_STEP_HOURS = float(time_step_hours)
        self.random_state = int(random_state)
        self.zone_df = zone_df.copy() if isinstance(zone_df, pd.DataFrame) else None
        self.operation_schedule_df = operation_schedule_df.copy() if isinstance(operation_schedule_df, pd.DataFrame) else None
        self.degradation_model = str(degradation_model)
        self.summary_only = bool(summary_only)
        self.max_memory_cache_entries = max(0, int(max_memory_cache_entries))
        self.base_weather, self.weather_meta = _load_base_weather(
            weather_mode=weather_mode,
            epw_path=epw_path,
            csv_path=csv_path,
            weather_df=weather_df,
            random_state=self.random_state,
            time_step_hours=self.base_cfg.TIME_STEP_HOURS,
        )
        self.cache: dict[tuple[Any, ...], dict[str, float]] = {}
        self.memory_cache_hits = 0
        self.disk_cache_hits = 0
        self.solver_evaluations = 0
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        if self.disk_cache_dir is not None:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # Include every base input and the actual weather values in the persistent
        # cache signature. This prevents a result from a previous setup being
        # reused after the user changes the building, configuration, zones,
        # schedule, weather source, analysis years, or time step.
        signature_payload = {
            "building": asdict(self.base_bldg),
            "config": asdict(self.base_cfg),
            "analysis_years": int(self.base_cfg.years),
            "random_state": self.random_state,
            "degradation_model": self.degradation_model,
            "zone_table": self.zone_df.to_dict(orient="records") if isinstance(self.zone_df, pd.DataFrame) else None,
            "operation_schedule": self.operation_schedule_df.to_dict(orient="records") if isinstance(self.operation_schedule_df, pd.DataFrame) else None,
            "weather_meta": self.weather_meta,
            "summary_only": self.summary_only,
        }
        h = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, default=str).encode("utf-8"))
        try:
            weather_hash = pd.util.hash_pandas_object(self.base_weather, index=True).values.tobytes()
            h.update(weather_hash)
        except Exception:
            h.update(str(self.base_weather.shape).encode("utf-8"))
        self.model_signature = h.hexdigest()

    def run(
        self,
        parameters: dict[str, float],
        strategy: str,
        severity: str,
        climate: str,
        seed: int,
        mode: str = "standard",
    ) -> dict[str, float]:
        cache_key = (
            tuple(sorted((str(k), round(float(v), 12)) for k, v in parameters.items())),
            strategy,
            severity,
            climate,
            int(seed),
            mode,
        )
        if cache_key in self.cache:
            self.memory_cache_hits += 1
            return dict(self.cache[cache_key])

        cache_file = None
        if self.disk_cache_dir is not None:
            cache_digest = hashlib.sha256(
                (self.model_signature + "|" + json.dumps(cache_key, sort_keys=True, default=str)).encode("utf-8")
            ).hexdigest()
            cache_file = self.disk_cache_dir / f"{cache_digest}.json"
            if cache_file.exists():
                try:
                    out = json.loads(cache_file.read_text(encoding="utf-8"))
                    self.disk_cache_hits += 1
                    if self.max_memory_cache_entries > 0:
                        self.cache[cache_key] = dict(out)
                        while len(self.cache) > self.max_memory_cache_entries:
                            self.cache.pop(next(iter(self.cache)))
                    return dict(out)
                except Exception:
                    # Ignore an incomplete/corrupted checkpoint and recompute it.
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass

        b_i = _clone_bldg(self.base_bldg)
        c_i = _clone_cfg(self.base_cfg)
        for name, raw_value in parameters.items():
            value = float(raw_value)
            if hasattr(b_i, name):
                setattr(b_i, name, value)
            elif hasattr(c_i, name):
                if name in {"FILTER_INTERVAL", "HX_INTERVAL", "APO_POP", "APO_ITERS"}:
                    value = int(round(value))
                setattr(c_i, name, value)
                if name in {"COP_COOL_NOM", "COP_HEAT_NOM", "FAN_EFF", "PUMP_SPECIFIC_W_M2", "AUXILIARY_W_M2"}:
                    c_i.USE_HVAC_PRESET = False
                    c_i.hvac_system_type = "Custom"
            else:
                raise AttributeError(f"Sensitivity parameter {name!r} is not present in BuildingSpec or HVACConfig.")

        if mode == "no_degradation":
            c_i.USE_DEGRADATION = False
            c_i.USE_MAINTENANCE_COST = False
        elif mode == "control_only":
            # Preserve physical degradation and S3 control, but make all S3 maintenance triggers unreachable.
            c_i.DEG_TRIGGER = 10.0
            c_i.RF_WARN = 1e9
            c_i.DP_WARN = 1e9
        elif mode == "maintenance_only":
            # Preserve S3 condition-based maintenance while fixing the control vector at nominal values.
            c_i.T_SP_MIN = c_i.T_SET
            c_i.T_SP_MAX = c_i.T_SET
            c_i.AF_MIN = 1.0
            c_i.AF_MAX = 1.0
        elif mode == "without_degradation_feedback":
            # Keep degradation physics and maintenance active but remove degradation from the S3 objective.
            remainder = max(c_i.W_ENERGY + c_i.W_COMFORT + c_i.W_CARBON, 1e-12)
            c_i.W_ENERGY = c_i.W_ENERGY / remainder
            c_i.W_COMFORT = c_i.W_COMFORT / remainder
            c_i.W_CARBON = c_i.W_CARBON / remainder
            c_i.W_DEGRAD = 0.0

        b_i, zone_meta = aggregate_zone_occupancy(b_i, self.zone_df)
        schedule_profile = zone_meta.get("schedule_profile", None)
        daily, annual, summary = simulate_combo(
            strategy=strategy,
            severity=severity,
            climate_name=climate,
            bldg=b_i,
            base_cfg=c_i,
            base_weather=self.base_weather,
            schedule_profile=schedule_profile,
            random_state=int(seed),
            degradation_model=self.degradation_model,
            operation_schedule_df=self.operation_schedule_df,
            summary_only=self.summary_only,
        )
        self.solver_evaluations += 1
        out = _standardize_summary(summary)
        out.update({"strategy": strategy, "severity": severity, "climate": climate, "mode": mode})
        del daily, annual, summary
        gc.collect()
        if self.max_memory_cache_entries > 0:
            self.cache[cache_key] = dict(out)
            while len(self.cache) > self.max_memory_cache_entries:
                self.cache.pop(next(iter(self.cache)))
        if cache_file is not None:
            tmp_file = cache_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
            os.replace(tmp_file, cache_file)
        return out

    def cache_stats(self) -> dict[str, int]:
        return {
            "solver_evaluations": int(self.solver_evaluations),
            "memory_cache_hits": int(self.memory_cache_hits),
            "disk_cache_hits": int(self.disk_cache_hits),
            "memory_cache_entries": int(len(self.cache)),
        }


def _new_run_dir(root: str | Path, analysis_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{_slug(analysis_name)}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    return path


def _write_metadata(run_dir: Path, metadata: dict[str, Any]) -> Path:
    p = run_dir / "metadata.json"
    p.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return p


def _zip_directory(run_dir: Path) -> Path:
    target = run_dir.with_suffix(".zip")
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in run_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(run_dir))
    return target


def _save_barh(df: pd.DataFrame, value_col: str, label_col: str, title: str, xlabel: str, png: Path, svg: Path, top_n: int = 20) -> None:
    show = df.sort_values(value_col, ascending=False).head(top_n).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9, max(4.8, 0.34 * len(show) + 1.8)))
    ax.barh(show[label_col], show[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def _save_oat_tornado(kpi_df: pd.DataFrame, kpi: str, png: Path, svg: Path) -> None:
    show = kpi_df.sort_values("max_abs_change_pct", ascending=False).head(20).sort_values("max_abs_change_pct")
    y = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(10, max(5.2, 0.38 * len(show) + 2.0)))
    ax.barh(y, show["low_change_pct"], label="Lower bound")
    ax.barh(y, show["high_change_pct"], label="Upper bound", alpha=0.70)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(show["label"])
    ax.set_xlabel("Change from baseline (%)")
    ax.set_title(f"Local OAT sensitivity — {KPI_LABELS.get(kpi, kpi)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def run_local_oat(
    runner: OfficialHVACRunner,
    specs: list[ParameterSpec],
    strategy: str,
    severity: str,
    climate: str,
    seed: int,
    output_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "local_OAT")
    baseline_params = {s.name: s.baseline for s in specs}
    baseline = runner.run(baseline_params, strategy, severity, climate, seed)
    case_rows = [{"case": "baseline", "parameter": "baseline", "label": "Baseline", "input_value": np.nan, **baseline}]
    total = 2 * len(specs)
    completed = 0
    for spec in specs:
        for case, value in (("low", spec.lower), ("high", spec.upper)):
            params = dict(baseline_params)
            params[spec.name] = value
            result = runner.run(params, strategy, severity, climate, seed)
            case_rows.append({"case": case, "parameter": spec.name, "label": spec.label, "group": spec.group, "input_value": value, **result})
            completed += 1
            if progress:
                progress(completed, total, f"{spec.label}: {case}")

    cases = pd.DataFrame(case_rows)
    cases_path = run_dir / "oat_all_cases.csv"
    cases.to_csv(cases_path, index=False)

    index_rows = []
    per_kpi_paths: dict[str, str] = {}
    for spec in specs:
        low = cases[(cases["parameter"] == spec.name) & (cases["case"] == "low")].iloc[0]
        high = cases[(cases["parameter"] == spec.name) & (cases["case"] == "high")].iloc[0]
        for kpi in STANDARD_KPIS:
            base_val = float(baseline.get(kpi, np.nan))
            low_val = float(low.get(kpi, np.nan))
            high_val = float(high.get(kpi, np.nan))
            denom = max(abs(base_val), 1e-12)
            low_pct = 100.0 * (low_val - base_val) / denom
            high_pct = 100.0 * (high_val - base_val) / denom
            rel_input_span = (spec.upper - spec.lower) / max(abs(spec.baseline), 1e-12)
            central_elasticity = ((high_val - low_val) / denom) / max(rel_input_span, 1e-12)
            index_rows.append({
                "parameter": spec.name,
                "label": spec.label,
                "group": spec.group,
                "kpi": kpi,
                "baseline_input": spec.baseline,
                "lower_input": spec.lower,
                "upper_input": spec.upper,
                "baseline_output": base_val,
                "low_output": low_val,
                "high_output": high_val,
                "low_change_pct": low_pct,
                "high_change_pct": high_pct,
                "max_abs_change_pct": max(abs(low_pct), abs(high_pct)),
                "central_elasticity": central_elasticity,
            })
    indices = pd.DataFrame(index_rows)
    indices_path = run_dir / "oat_indices_all_kpis.csv"
    indices.to_csv(indices_path, index=False)
    for kpi in STANDARD_KPIS:
        kdf = indices[indices["kpi"] == kpi].copy()
        csv_path = run_dir / f"oat_{_slug(kpi)}.csv"
        png = run_dir / "figures" / f"oat_tornado_{_slug(kpi)}.png"
        svg = run_dir / "figures" / f"oat_tornado_{_slug(kpi)}.svg"
        kdf.to_csv(csv_path, index=False)
        _save_oat_tornado(kdf, kpi, png, svg)
        per_kpi_paths[f"{kpi}_csv"] = str(csv_path)
        per_kpi_paths[f"{kpi}_png"] = str(png)
        per_kpi_paths[f"{kpi}_svg"] = str(svg)

    metadata = _write_metadata(run_dir, {
        "method": "signed one-at-a-time lower/baseline/upper sensitivity",
        "strategy": strategy,
        "severity": severity,
        "climate": climate,
        "seed": seed,
        "n_parameters": len(specs),
        "weather": runner.weather_meta,
        "chillerplant_csv_used": False,
    })
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "all_cases_csv": str(cases_path), "indices_csv": str(indices_path), "metadata_json": str(metadata), "zip": str(zip_path), **per_kpi_paths}


def _draw_samples(specs: list[ParameterSpec], n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, float]]]:
    unit = rng.random((n, len(specs)))
    rows = [_params_from_unit(specs, unit[i]) for i in range(n)]
    return unit, rows


def run_paired_monte_carlo(
    runner: OfficialHVACRunner,
    specs: list[ParameterSpec],
    strategies: list[str],
    severity: str,
    climate: str,
    seed: int,
    n_samples: int,
    output_root: str | Path,
    fixed_optimizer_seed: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "paired_Monte_Carlo")
    rng = np.random.default_rng(seed)
    _, param_rows = _draw_samples(specs, int(n_samples), rng)
    records = []
    total = len(param_rows) * len(strategies)
    done = 0
    for sample_id, params in enumerate(param_rows, start=1):
        for strategy in strategies:
            run_seed = seed if fixed_optimizer_seed else seed + sample_id
            result = runner.run(params, strategy, severity, climate, run_seed)
            records.append({"sample_id": sample_id, **params, **result})
            done += 1
            if progress:
                progress(done, total, f"Sample {sample_id}/{n_samples}, {strategy}")
    raw = pd.DataFrame(records)
    raw_path = run_dir / "monte_carlo_paired_samples.csv"
    raw.to_csv(raw_path, index=False)

    summary_rows = []
    for strategy, sdf in raw.groupby("strategy"):
        for kpi in STANDARD_KPIS:
            vals = pd.to_numeric(sdf[kpi], errors="coerce").dropna()
            if vals.empty:
                continue
            summary_rows.append({
                "strategy": strategy,
                "kpi": kpi,
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                "cv_pct": 100.0 * vals.std(ddof=1) / max(abs(vals.mean()), 1e-12) if len(vals) > 1 else 0.0,
                "p05": vals.quantile(0.05),
                "p25": vals.quantile(0.25),
                "p50": vals.quantile(0.50),
                "p75": vals.quantile(0.75),
                "p95": vals.quantile(0.95),
                "min": vals.min(),
                "max": vals.max(),
            })
    summary = pd.DataFrame(summary_rows)
    summary_path = run_dir / "monte_carlo_summary.csv"
    summary.to_csv(summary_path, index=False)

    ranking_rows = []
    for kpi in ["total_energy_MWh", "total_cost_USD", "co2_tonne", "mean_comfort_dev_C", "mean_residual_delta"]:
        winners = raw.loc[raw.groupby("sample_id")[kpi].idxmin(), ["sample_id", "strategy", kpi]]
        probs = winners["strategy"].value_counts(normalize=True)
        for strategy in strategies:
            ranking_rows.append({"kpi": kpi, "strategy": strategy, "probability_ranked_best": float(probs.get(strategy, 0.0))})
    ranking = pd.DataFrame(ranking_rows)
    ranking_path = run_dir / "strategy_ranking_probabilities.csv"
    ranking.to_csv(ranking_path, index=False)

    paired_rows = []
    if {"S0", "S3"}.issubset(set(strategies)):
        for kpi in STANDARD_KPIS:
            wide = raw.pivot(index="sample_id", columns="strategy", values=kpi)
            if "S0" not in wide.columns or "S3" not in wide.columns:
                continue
            delta = wide["S0"] - wide["S3"]
            pct = 100.0 * delta / wide["S0"].abs().replace(0.0, np.nan)
            paired_rows.append({
                "kpi": kpi,
                "probability_S3_lower_than_S0": float((wide["S3"] < wide["S0"]).mean()),
                "median_S0_minus_S3": float(delta.median()),
                "median_S3_improvement_pct": float(pct.median()),
                "p05_improvement_pct": float(pct.quantile(0.05)),
                "p95_improvement_pct": float(pct.quantile(0.95)),
            })
    paired = pd.DataFrame(paired_rows)
    paired_path = run_dir / "paired_S3_vs_S0_robustness.csv"
    paired.to_csv(paired_path, index=False)

    figure_paths = {}
    for kpi in STANDARD_KPIS:
        arrays = []
        labels = []
        for strategy in strategies:
            vals = pd.to_numeric(raw.loc[raw["strategy"] == strategy, kpi], errors="coerce").dropna().to_numpy()
            if len(vals):
                arrays.append(vals)
                labels.append(strategy)
        if not arrays:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5.3))
        ax.boxplot(arrays, labels=labels, showmeans=True)
        ax.set_title(f"Paired uncertainty analysis — {KPI_LABELS.get(kpi, kpi)}")
        ax.set_ylabel(KPI_LABELS.get(kpi, kpi))
        fig.tight_layout()
        png = run_dir / "figures" / f"monte_carlo_boxplot_{_slug(kpi)}.png"
        svg = run_dir / "figures" / f"monte_carlo_boxplot_{_slug(kpi)}.svg"
        fig.savefig(png, dpi=400, bbox_inches="tight")
        fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        figure_paths[f"{kpi}_png"] = str(png)
        figure_paths[f"{kpi}_svg"] = str(svg)

    metadata = _write_metadata(run_dir, {
        "method": "paired Monte Carlo uncertainty and strategy-ranking robustness",
        "n_samples": n_samples,
        "strategies": strategies,
        "severity": severity,
        "climate": climate,
        "fixed_optimizer_seed": fixed_optimizer_seed,
        "seed": seed,
        "weather": runner.weather_meta,
        "chillerplant_csv_used": False,
    })
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "samples_csv": str(raw_path), "summary_csv": str(summary_path), "ranking_csv": str(ranking_path), "paired_csv": str(paired_path), "metadata_json": str(metadata), "zip": str(zip_path), **figure_paths}


def run_morris(
    runner: OfficialHVACRunner,
    specs: list[ParameterSpec],
    strategy: str,
    severity: str,
    climate: str,
    seed: int,
    trajectories: int,
    levels: int,
    kpi: str,
    output_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "Morris_global_screening")
    rng = np.random.default_rng(seed)
    k = len(specs)
    p = max(4, int(levels))
    delta = p / (2.0 * (p - 1.0))
    baseline_result = runner.run({s.name: s.baseline for s in specs}, strategy, severity, climate, seed)
    scale_y = max(abs(float(baseline_result[kpi])), 1e-12)
    effect_rows = []
    sample_rows = []
    total = int(trajectories) * (k + 1)
    done = 0
    for trajectory in range(1, int(trajectories) + 1):
        directions = rng.choice([-1.0, 1.0], size=k)
        x = np.empty(k)
        for j, direction in enumerate(directions):
            x[j] = rng.uniform(0.0, 1.0 - delta) if direction > 0 else rng.uniform(delta, 1.0)
        order = rng.permutation(k)
        params = _params_from_unit(specs, x)
        y_prev = float(runner.run(params, strategy, severity, climate, seed).get(kpi, np.nan))
        sample_rows.append({"trajectory": trajectory, "step": 0, **params, kpi: y_prev})
        done += 1
        if progress:
            progress(done, total, f"Trajectory {trajectory}, start")
        for step_no, j in enumerate(order, start=1):
            x_new = x.copy()
            x_new[j] += directions[j] * delta
            params_new = _params_from_unit(specs, x_new)
            y_new = float(runner.run(params_new, strategy, severity, climate, seed).get(kpi, np.nan))
            ee = (y_new - y_prev) / (directions[j] * delta)
            effect_rows.append({
                "trajectory": trajectory,
                "step": step_no,
                "parameter": specs[j].name,
                "label": specs[j].label,
                "group": specs[j].group,
                "elementary_effect": ee,
                "normalized_elementary_effect": ee / scale_y,
                "output_before": y_prev,
                "output_after": y_new,
            })
            sample_rows.append({"trajectory": trajectory, "step": step_no, **params_new, kpi: y_new})
            x = x_new
            y_prev = y_new
            done += 1
            if progress:
                progress(done, total, f"Trajectory {trajectory}, {specs[j].label}")
    effects = pd.DataFrame(effect_rows)
    samples = pd.DataFrame(sample_rows)
    indices = effects.groupby(["parameter", "label", "group"], as_index=False).agg(
        mu=("elementary_effect", "mean"),
        mu_star=("elementary_effect", lambda s: float(np.mean(np.abs(s)))),
        sigma=("elementary_effect", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
        normalized_mu=("normalized_elementary_effect", "mean"),
        normalized_mu_star=("normalized_elementary_effect", lambda s: float(np.mean(np.abs(s)))),
        normalized_sigma=("normalized_elementary_effect", lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else 0.0),
    ).sort_values("normalized_mu_star", ascending=False)
    effects_path = run_dir / "morris_elementary_effects.csv"
    indices_path = run_dir / "morris_indices.csv"
    samples_path = run_dir / "morris_samples.csv"
    effects.to_csv(effects_path, index=False)
    indices.to_csv(indices_path, index=False)
    samples.to_csv(samples_path, index=False)
    png = run_dir / "figures" / f"morris_ranking_{_slug(kpi)}.png"
    svg = run_dir / "figures" / f"morris_ranking_{_slug(kpi)}.svg"
    _save_barh(indices, "normalized_mu_star", "label", f"Morris screening — {KPI_LABELS.get(kpi, kpi)}", "Normalized μ*", png, svg)
    scatter_png = run_dir / "figures" / f"morris_mu_star_sigma_{_slug(kpi)}.png"
    scatter_svg = run_dir / "figures" / f"morris_mu_star_sigma_{_slug(kpi)}.svg"
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.scatter(indices["normalized_mu_star"], indices["normalized_sigma"])
    for _, row in indices.iterrows():
        ax.annotate(row["label"], (row["normalized_mu_star"], row["normalized_sigma"]), fontsize=8)
    ax.set_xlabel("Normalized μ*")
    ax.set_ylabel("Normalized σ")
    ax.set_title(f"Morris nonlinearity/interaction map — {KPI_LABELS.get(kpi, kpi)}")
    fig.tight_layout()
    fig.savefig(scatter_png, dpi=400, bbox_inches="tight")
    fig.savefig(scatter_svg, bbox_inches="tight")
    plt.close(fig)
    metadata = _write_metadata(run_dir, {
        "method": "Morris elementary-effects global screening",
        "trajectories": trajectories,
        "levels": levels,
        "delta_in_unit_space": delta,
        "kpi": kpi,
        "strategy": strategy,
        "severity": severity,
        "climate": climate,
        "seed": seed,
        "weather": runner.weather_meta,
        "chillerplant_csv_used": False,
    })
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "indices_csv": str(indices_path), "effects_csv": str(effects_path), "samples_csv": str(samples_path), "ranking_png": str(png), "ranking_svg": str(svg), "map_png": str(scatter_png), "map_svg": str(scatter_svg), "metadata_json": str(metadata), "zip": str(zip_path)}


def _sobol_indices_from_arrays(ya: np.ndarray, yb: np.ndarray, yabi: np.ndarray) -> tuple[float, float]:
    var_y = float(np.var(np.concatenate([ya, yb]), ddof=1))
    if not np.isfinite(var_y) or var_y <= 1e-18:
        return np.nan, np.nan
    s1 = float(np.mean(yb * (yabi - ya)) / var_y)
    st_idx = float(0.5 * np.mean((ya - yabi) ** 2) / var_y)
    return s1, st_idx


def run_sobol(
    runner: OfficialHVACRunner,
    specs: list[ParameterSpec],
    strategy: str,
    severity: str,
    climate: str,
    seed: int,
    base_n: int,
    kpi: str,
    bootstrap_reps: int,
    output_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "Sobol_global_sensitivity")
    rng = np.random.default_rng(seed)
    n = int(base_n)
    k = len(specs)
    a = rng.random((n, k))
    b = rng.random((n, k))
    ya = np.empty(n)
    yb = np.empty(n)
    yab = np.empty((k, n))
    raw_rows = []
    total = n * (k + 2)
    done = 0
    for r in range(n):
        pa = _params_from_unit(specs, a[r])
        pb = _params_from_unit(specs, b[r])
        ya[r] = runner.run(pa, strategy, severity, climate, seed).get(kpi, np.nan)
        yb[r] = runner.run(pb, strategy, severity, climate, seed).get(kpi, np.nan)
        raw_rows.append({"matrix": "A", "sample": r + 1, **pa, kpi: ya[r]})
        raw_rows.append({"matrix": "B", "sample": r + 1, **pb, kpi: yb[r]})
        done += 2
        if progress:
            progress(done, total, f"Base sample {r + 1}/{n}")
    index_rows = []
    for i, spec in enumerate(specs):
        ab_i = a.copy()
        ab_i[:, i] = b[:, i]
        for r in range(n):
            params = _params_from_unit(specs, ab_i[r])
            yab[i, r] = runner.run(params, strategy, severity, climate, seed).get(kpi, np.nan)
            raw_rows.append({"matrix": f"AB_{spec.name}", "sample": r + 1, **params, kpi: yab[i, r]})
            done += 1
            if progress:
                progress(done, total, f"Hybrid {spec.label}, {r + 1}/{n}")
        s1, st_idx = _sobol_indices_from_arrays(ya, yb, yab[i])
        s1_boot = []
        st_boot = []
        for _ in range(int(max(0, bootstrap_reps))):
            idx = rng.integers(0, n, size=n)
            bs1, bst = _sobol_indices_from_arrays(ya[idx], yb[idx], yab[i, idx])
            if np.isfinite(bs1):
                s1_boot.append(bs1)
            if np.isfinite(bst):
                st_boot.append(bst)
        index_rows.append({
            "parameter": spec.name,
            "label": spec.label,
            "group": spec.group,
            "S1": s1,
            "ST": st_idx,
            "interaction_gap_ST_minus_S1": st_idx - s1 if np.isfinite(s1) and np.isfinite(st_idx) else np.nan,
            "S1_ci_low": np.quantile(s1_boot, 0.025) if s1_boot else np.nan,
            "S1_ci_high": np.quantile(s1_boot, 0.975) if s1_boot else np.nan,
            "ST_ci_low": np.quantile(st_boot, 0.025) if st_boot else np.nan,
            "ST_ci_high": np.quantile(st_boot, 0.975) if st_boot else np.nan,
        })
    indices = pd.DataFrame(index_rows).sort_values("ST", ascending=False)
    raw = pd.DataFrame(raw_rows)
    indices_path = run_dir / "sobol_indices.csv"
    raw_path = run_dir / "sobol_samples.csv"
    indices.to_csv(indices_path, index=False)
    raw.to_csv(raw_path, index=False)

    png = run_dir / "figures" / f"sobol_indices_{_slug(kpi)}.png"
    svg = run_dir / "figures" / f"sobol_indices_{_slug(kpi)}.svg"
    show = indices.sort_values("ST").tail(20)
    y = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.38 * len(show) + 2.0)))
    ax.barh(y - 0.18, show["S1"], height=0.36, label="First order S1")
    ax.barh(y + 0.18, show["ST"], height=0.36, label="Total order ST")
    ax.set_yticks(y)
    ax.set_yticklabels(show["label"])
    ax.set_xlabel("Sobol index")
    ax.set_title(f"Sobol sensitivity — {KPI_LABELS.get(kpi, kpi)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    metadata = _write_metadata(run_dir, {
        "method": "Saltelli first-order and Jansen total-order Monte Carlo estimators",
        "base_sample_size": n,
        "model_evaluations": int(n * (k + 2)),
        "bootstrap_repetitions": bootstrap_reps,
        "kpi": kpi,
        "strategy": strategy,
        "severity": severity,
        "climate": climate,
        "seed": seed,
        "note": "Finite-sample estimates may be negative or exceed one; increase N rather than clipping.",
        "weather": runner.weather_meta,
        "chillerplant_csv_used": False,
    })
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "indices_csv": str(indices_path), "samples_csv": str(raw_path), "figure_png": str(png), "figure_svg": str(svg), "metadata_json": str(metadata), "zip": str(zip_path)}


def run_benchmarks(
    runner: OfficialHVACRunner,
    baseline_params: dict[str, float],
    strategies: list[str],
    nominal_severity: str,
    climate: str,
    seed: int,
    output_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "benchmark_cases")
    cases = [
        ("No degradation", nominal_severity, "no_degradation", {}),
        ("Nominal", nominal_severity, "standard", {}),
        ("Severe degradation", "Severe", "standard", {}),
        ("High infiltration", nominal_severity, "standard", {"infiltration_ach": runner.base_bldg.infiltration_ach * 1.30}),
    ]
    records = []
    total = len(cases) * len(strategies)
    done = 0
    for case_name, severity, mode, overrides in cases:
        params = dict(baseline_params)
        params.update(overrides)
        for strategy in strategies:
            result = runner.run(params, strategy, severity, climate, seed, mode=mode)
            records.append({"benchmark_case": case_name, **result})
            done += 1
            if progress:
                progress(done, total, f"{case_name}: {strategy}")
    df = pd.DataFrame(records)
    path = run_dir / "benchmark_results.csv"
    df.to_csv(path, index=False)
    wide = df.pivot(index="benchmark_case", columns="strategy", values="total_energy_MWh")
    if "S0" in wide.columns:
        saving_rows = []
        for case_name, row in wide.iterrows():
            for strategy in strategies:
                if strategy in row.index:
                    saving_rows.append({"benchmark_case": case_name, "strategy": strategy, "energy_saving_vs_S0_pct": 100.0 * (row["S0"] - row[strategy]) / max(abs(row["S0"]), 1e-12)})
        savings = pd.DataFrame(saving_rows)
    else:
        savings = pd.DataFrame()
    savings_path = run_dir / "benchmark_energy_savings.csv"
    savings.to_csv(savings_path, index=False)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    wide.plot(kind="bar", ax=ax)
    ax.set_ylabel("Total energy (MWh)")
    ax.set_title("Benchmark cases across S0–S3")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    png = run_dir / "figures" / "benchmark_energy.png"
    svg = run_dir / "figures" / "benchmark_energy.svg"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    metadata = _write_metadata(run_dir, {"method": "benchmark stress cases", "strategies": strategies, "climate": climate, "seed": seed, "weather": runner.weather_meta, "chillerplant_csv_used": False})
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "results_csv": str(path), "savings_csv": str(savings_path), "figure_png": str(png), "figure_svg": str(svg), "metadata_json": str(metadata), "zip": str(zip_path)}


def run_ablation(
    runner: OfficialHVACRunner,
    baseline_params: dict[str, float],
    severity: str,
    climate: str,
    seed: int,
    output_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "S3_ablation")
    cases = [
        ("S0 reactive baseline", "S0", "standard"),
        ("Full S3", "S3", "standard"),
        ("S3 control only", "S3", "control_only"),
        ("S3 maintenance only", "S3", "maintenance_only"),
        ("S3 without degradation feedback", "S3", "without_degradation_feedback"),
    ]
    records = []
    for i, (name, strategy, mode) in enumerate(cases, start=1):
        result = runner.run(baseline_params, strategy, severity, climate, seed, mode=mode)
        records.append({"ablation_case": name, **result})
        if progress:
            progress(i, len(cases), name)
    df = pd.DataFrame(records)
    s0_energy = float(df.loc[df["ablation_case"] == "S0 reactive baseline", "total_energy_MWh"].iloc[0])
    df["energy_saving_vs_S0_pct"] = 100.0 * (s0_energy - df["total_energy_MWh"]) / max(abs(s0_energy), 1e-12)
    path = run_dir / "ablation_results.csv"
    df.to_csv(path, index=False)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(df["ablation_case"], df["total_energy_MWh"])
    ax.set_ylabel("Total energy (MWh)")
    ax.set_title("S3 mechanism ablation")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    png = run_dir / "figures" / "ablation_energy.png"
    svg = run_dir / "figures" / "ablation_energy.svg"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    metadata = _write_metadata(run_dir, {
        "method": "mechanism ablation",
        "severity": severity,
        "climate": climate,
        "seed": seed,
        "control_only_definition": "S3 optimization active with maintenance thresholds made unreachable.",
        "maintenance_only_definition": "S3 maintenance active with setpoint and airflow fixed at nominal values.",
        "without_degradation_feedback_definition": "Physical degradation and maintenance active; W_DEGRAD set to zero and remaining objective weights renormalized.",
        "weather": runner.weather_meta,
        "chillerplant_csv_used": False,
    })
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "results_csv": str(path), "figure_png": str(png), "figure_svg": str(svg), "metadata_json": str(metadata), "zip": str(zip_path)}


def run_optimizer_robustness(
    runner: OfficialHVACRunner,
    baseline_params: dict[str, float],
    severity: str,
    climate: str,
    seed_values: list[int],
    populations: list[int],
    iterations: list[int],
    output_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "S3_optimizer_robustness")
    rows = []
    total = len(seed_values) * len(populations) * len(iterations)
    done = 0
    for pop in populations:
        for iters in iterations:
            for seed in seed_values:
                params = dict(baseline_params)
                params["APO_POP"] = int(pop)
                params["APO_ITERS"] = int(iters)
                result = runner.run(params, "S3", severity, climate, int(seed))
                rows.append({"optimizer_population": pop, "optimizer_iterations": iters, "seed": seed, "candidate_evaluations_per_step": pop * iters, **result})
                done += 1
                if progress:
                    progress(done, total, f"N={pop}, K={iters}, seed={seed}")
    raw = pd.DataFrame(rows)
    raw_path = run_dir / "optimizer_robustness_raw.csv"
    raw.to_csv(raw_path, index=False)
    summary = raw.groupby(["optimizer_population", "optimizer_iterations", "candidate_evaluations_per_step"], as_index=False).agg(
        n=("seed", "count"),
        energy_mean_MWh=("total_energy_MWh", "mean"),
        energy_std_MWh=("total_energy_MWh", "std"),
        energy_min_MWh=("total_energy_MWh", "min"),
        energy_max_MWh=("total_energy_MWh", "max"),
        cost_mean_USD=("total_cost_USD", "mean"),
        comfort_mean_C=("mean_comfort_dev_C", "mean"),
        degradation_mean=("mean_residual_delta", "mean"),
    )
    summary["energy_cv_pct"] = 100.0 * summary["energy_std_MWh"] / summary["energy_mean_MWh"].abs().replace(0.0, np.nan)
    summary_path = run_dir / "optimizer_robustness_summary.csv"
    summary.to_csv(summary_path, index=False)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for pop, sdf in summary.groupby("optimizer_population"):
        ax.plot(sdf["optimizer_iterations"], sdf["energy_mean_MWh"], marker="o", label=f"Population {pop}")
    ax.set_xlabel("Optimizer iterations")
    ax.set_ylabel("Mean total energy (MWh)")
    ax.set_title("S3 optimizer convergence/robustness")
    ax.legend()
    fig.tight_layout()
    png = run_dir / "figures" / "optimizer_robustness_energy.png"
    svg = run_dir / "figures" / "optimizer_robustness_energy.svg"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    metadata = _write_metadata(run_dir, {"method": "population–iteration–seed robustness", "severity": severity, "climate": climate, "populations": populations, "iterations": iterations, "seeds": seed_values, "weather": runner.weather_meta, "chillerplant_csv_used": False})
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "raw_csv": str(raw_path), "summary_csv": str(summary_path), "figure_png": str(png), "figure_svg": str(svg), "metadata_json": str(metadata), "zip": str(zip_path)}


def run_connectivity_test(
    runner: OfficialHVACRunner,
    specs: list[ParameterSpec],
    strategy: str,
    severity: str,
    climate: str,
    seed: int,
    output_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, str]:
    run_dir = _new_run_dir(output_root, "parameter_connectivity")
    baseline_params = {s.name: s.baseline for s in specs}
    base = runner.run(baseline_params, strategy, severity, climate, seed)
    rows = []
    for i, spec in enumerate(specs, start=1):
        params = dict(baseline_params)
        params[spec.name] = spec.upper
        result = runner.run(params, strategy, severity, climate, seed)
        row = {"parameter": spec.name, "label": spec.label, "group": spec.group, "baseline_input": spec.baseline, "test_input": spec.upper}
        max_change = 0.0
        for kpi in STANDARD_KPIS:
            delta = float(result[kpi]) - float(base[kpi])
            pct = 100.0 * delta / max(abs(float(base[kpi])), 1e-12)
            row[f"delta_{kpi}"] = delta
            row[f"change_pct_{kpi}"] = pct
            max_change = max(max_change, abs(pct))
        row["max_abs_kpi_change_pct"] = max_change
        row["connected"] = bool(max_change > 1e-8)
        rows.append(row)
        if progress:
            progress(i, len(specs), spec.label)
    df = pd.DataFrame(rows).sort_values("max_abs_kpi_change_pct", ascending=False)
    path = run_dir / "parameter_connectivity_test.csv"
    df.to_csv(path, index=False)
    png = run_dir / "figures" / "parameter_connectivity.png"
    svg = run_dir / "figures" / "parameter_connectivity.svg"
    _save_barh(df, "max_abs_kpi_change_pct", "label", "Parameter connectivity to official solver outputs", "Maximum absolute KPI change (%)", png, svg)
    metadata = _write_metadata(run_dir, {"method": "upper-bound parameter connectivity diagnostic", "strategy": strategy, "severity": severity, "climate": climate, "seed": seed, "weather": runner.weather_meta, "chillerplant_csv_used": False})
    zip_path = _zip_directory(run_dir)
    return {"run_dir": str(run_dir), "results_csv": str(path), "figure_png": str(png), "figure_svg": str(svg), "metadata_json": str(metadata), "zip": str(zip_path)}



def _attach_runner_cache_stats(paths: dict[str, str], runner: OfficialHVACRunner) -> dict[str, str]:
    """Persist cache/resume statistics beside each completed analysis."""
    try:
        run_dir = Path(paths.get("run_dir", ""))
        if run_dir.exists():
            stats_path = run_dir / "evaluation_cache_stats.json"
            stats_path.write_text(json.dumps(runner.cache_stats(), indent=2), encoding="utf-8")
            paths = dict(paths)
            paths["cache_stats_json"] = str(stats_path)
    except Exception:
        pass
    return paths

def _render_progress() -> tuple[Any, Any, Callable[[int, int, str], None]]:
    bar = st.progress(0.0)
    status = st.empty()

    def update(done: int, total: int, message: str):
        bar.progress(min(max(done / max(total, 1), 0.0), 1.0))
        status.caption(f"{done:,}/{total:,} — {message}")

    return bar, status, update


def _show_paths(paths: dict[str, str], prefix: str) -> None:
    existing = []
    for key, value in paths.items():
        if key == "run_dir":
            continue
        p = Path(str(value))
        if p.exists() and p.is_file():
            existing.append((key, p))
    if not existing:
        return
    st.success(f"Results written to: {paths.get('run_dir', '')}")
    cols = st.columns(3)
    for i, (key, p) in enumerate(existing):
        with cols[i % 3]:
            try:
                size_mb = p.stat().st_size / (1024 * 1024)
                if size_mb <= 100:
                    st.download_button(
                        label=f"Download {key}",
                        data=p.read_bytes(),
                        file_name=p.name,
                        key=f"{prefix}_{key}_{p.stat().st_mtime_ns}",
                    )
                else:
                    st.caption(f"{p.name}: {size_mb:.1f} MB; download from deployment storage.")
            except Exception as exc:
                st.caption(f"Could not prepare {p.name}: {exc}")


def _preview_csv(path: str | None, rows: int = 250) -> None:
    if not path:
        return
    p = Path(path)
    if p.exists() and p.suffix.lower() == ".csv":
        try:
            st.dataframe(pd.read_csv(p, nrows=rows), width="stretch")
        except Exception as exc:
            st.warning(f"Preview failed: {exc}")


def _parse_int_list(text: str, minimum: int = 1) -> list[int]:
    vals = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if token:
            vals.append(max(minimum, int(token)))
    return sorted(set(vals))


def render_integrated_sensitivity_tab(
    bldg: BuildingSpec,
    cfg: HVACConfig,
    build_weather_controls: Callable[[str], tuple[Any, ...]],
    time_step_hours: float,
    zone_df: pd.DataFrame | None = None,
    operation_schedule_df: pd.DataFrame | None = None,
) -> None:
    if st is None:
        raise ImportError("Streamlit is required to render the sensitivity studio UI.")
    st.subheader("Comprehensive Local and Global Sensitivity Studio")
    st.markdown(
        """
        This tab uses the **official HVAC v3 dynamic RC solver** for every evaluation. It performs signed local OAT sensitivity,
        paired Monte Carlo robustness, Morris screening, Sobol first/total-order analysis, benchmark cases, S3 ablation,
        optimizer robustness, and parameter-connectivity diagnostics. **ChillerPlant.csv is not used.**
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    analysis_years = int(c1.number_input("Sensitivity analysis years", 1, 10, 1, 1, key="igs_years"))
    strategy = c2.selectbox("Primary strategy", list(SCENARIOS.keys()), index=3, key="igs_strategy")
    severity = c3.selectbox("Primary degradation severity", list(SEVERITY_LEVELS.keys()), index=1, key="igs_severity")
    climate = c4.selectbox("Primary climate", list(CLIMATE_LEVELS.keys()), index=0, key="igs_climate")

    st.markdown("### Cloud-safe calculation controls")
    c1, c2, c3 = st.columns(3)
    sensitivity_step_label = c1.selectbox(
        "Sensitivity solver time step",
        ["Daily (24 h) — recommended for cloud", "Use main model time step", "12-hour", "6-hour", "3-hour", "Hourly"],
        index=0,
        key="igs_solver_time_step",
        help="This applies only to sensitivity evaluations. Daily mode greatly reduces cloud CPU and memory use. Use the main/hourly step for final high-resolution runs on a local workstation.",
    )
    if sensitivity_step_label == "Use main model time step":
        sensitivity_time_step_hours = float(time_step_hours)
    else:
        sensitivity_time_step_hours = {
            "Daily (24 h) — recommended for cloud": 24.0,
            "12-hour": 12.0,
            "6-hour": 6.0,
            "3-hour": 3.0,
            "Hourly": 1.0,
        }[sensitivity_step_label]

    optimizer_profile = c2.selectbox(
        "S3 optimizer workload",
        ["Current model settings", "Cloud screening (8 × 4)", "Fast diagnostic (4 × 2)"],
        index=0,
        key="igs_optimizer_profile",
        help="Reduced profiles are for software checks and preliminary screening. Final manuscript results should use the validated current-model optimizer settings.",
    )
    sensitivity_cfg = _clone_cfg(cfg)
    if optimizer_profile == "Cloud screening (8 × 4)":
        sensitivity_cfg.APO_POP = 8
        sensitivity_cfg.APO_ITERS = 4
    elif optimizer_profile == "Fast diagnostic (4 × 2)":
        sensitivity_cfg.APO_POP = 4
        sensitivity_cfg.APO_ITERS = 2
    c3.metric("S3 candidates per timestep", f"{int(sensitivity_cfg.APO_POP) * int(sensitivity_cfg.APO_ITERS):,}")

    weather_mode, epw_path, csv_path, weather_df, seed, base_output_root = build_weather_controls("integrated_sensitivity")
    # `build_weather_controls()` creates a text_input widget whose key is
    # `integrated_sensitivity_out_dir`. Streamlit forbids assigning to that same
    # session-state key after the widget has been instantiated. Keep the widget
    # value unchanged and store the derived analysis folder under a separate key.
    output_root = str(Path(base_output_root) / "comprehensive_sensitivity")
    st.session_state["integrated_sensitivity_results_dir"] = output_root
    evaluation_cache_dir = Path(output_root) / "_evaluation_cache"
    evaluation_cache_dir.mkdir(parents=True, exist_ok=True)

    c1, c2 = st.columns([3, 1])
    c1.info(
        "Cloud-safe mode is active: sensitivity evaluations use the official solver in summary-only memory mode. "
        "Each completed evaluation is checkpointed to disk. If the app is interrupted, press the same Run button again; "
        "completed cases will be restored and only missing cases will be calculated."
    )
    if c2.button("Clear sensitivity checkpoints", key="igs_clear_checkpoints"):
        shutil.rmtree(evaluation_cache_dir, ignore_errors=True)
        evaluation_cache_dir.mkdir(parents=True, exist_ok=True)
        st.success("Sensitivity evaluation checkpoints cleared.")
    checkpoint_count = sum(1 for _ in evaluation_cache_dir.glob("*.json"))
    st.caption(
        f"Checkpoint directory: {evaluation_cache_dir} | cached evaluations: {checkpoint_count:,} | "
        f"sensitivity step: {sensitivity_time_step_hours:g} h | optimizer: {int(sensitivity_cfg.APO_POP)} × {int(sensitivity_cfg.APO_ITERS)}"
    )

    st.markdown("### Uncertain-parameter ranges")
    defaults = specs_to_dataframe(default_parameter_specs(bldg, cfg))
    if "igs_parameter_table" not in st.session_state:
        st.session_state["igs_parameter_table"] = defaults
    c1, c2 = st.columns([1, 1])
    if c1.button("Reset parameter ranges from current setup", key="igs_reset_ranges"):
        st.session_state["igs_parameter_table"] = defaults
    uploaded_ranges = c2.file_uploader("Upload parameter ranges CSV", type=["csv"], key="igs_ranges_upload")
    if uploaded_ranges is not None:
        st.session_state["igs_parameter_table"] = pd.read_csv(uploaded_ranges)
    edited = st.data_editor(st.session_state["igs_parameter_table"], num_rows="dynamic", width="stretch", key="igs_ranges_editor")
    st.session_state["igs_parameter_table"] = edited
    st.download_button("Download current parameter ranges", edited.to_csv(index=False).encode("utf-8"), file_name="sensitivity_parameter_ranges.csv", mime="text/csv", key="igs_download_ranges")

    try:
        specs_all = specs_from_dataframe(edited)
    except Exception as exc:
        st.error(f"Parameter-range error: {exc}")
        return

    c1, c2 = st.columns(2)
    use_zone = c1.checkbox("Use current zone-specific table", value=isinstance(zone_df, pd.DataFrame), key="igs_use_zone")
    use_schedule = c2.checkbox("Use current operation schedule/EMS", value=bool(getattr(cfg, "EMS_CUSTOM_SCHEDULE_ENABLED", False)), key="igs_use_schedule")
    active_zone_df = zone_df if use_zone else None
    active_schedule = operation_schedule_df if use_schedule else None

    if weather_mode == "uploaded" and weather_df is None:
        st.warning("Upload a weather file or select synthetic/path weather before running an analysis.")

    def make_runner() -> OfficialHVACRunner:
        if weather_mode == "uploaded" and weather_df is None:
            raise ValueError("No uploaded weather data are available.")
        return OfficialHVACRunner(
            bldg=bldg,
            cfg=sensitivity_cfg,
            analysis_years=analysis_years,
            weather_mode=weather_mode,
            epw_path=epw_path,
            csv_path=csv_path,
            weather_df=weather_df,
            random_state=int(seed),
            zone_df=active_zone_df,
            operation_schedule_df=active_schedule,
            degradation_model=str(getattr(sensitivity_cfg, "degradation_model", "physics")),
            time_step_hours=float(sensitivity_time_step_hours),
            disk_cache_dir=evaluation_cache_dir,
            summary_only=True,
            max_memory_cache_entries=128,
        )

    sub = st.tabs(["Local OAT", "Monte Carlo", "Morris", "Sobol", "Benchmarks", "Ablation", "Optimizer", "Connectivity", "Export Index"])

    with sub[0]:
        st.markdown("#### Signed one-at-a-time local sensitivity")
        st.caption("Runs baseline, lower-bound, and upper-bound cases for every enabled parameter. Each KPI receives a separate CSV and tornado figure.")
        if st.button("Run Local OAT analysis", type="primary", key="igs_run_oat"):
            try:
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_local_oat(runner, specs_all, strategy, severity, climate, int(seed), output_root, callback), runner)
                bar.progress(1.0); status.caption("Local OAT completed.")
                st.session_state["igs_oat_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_oat_paths", {})
        if paths:
            _preview_csv(paths.get("indices_csv"))
            _show_paths(paths, "igs_oat")

    with sub[1]:
        st.markdown("#### Paired Monte Carlo uncertainty and strategy-ranking robustness")
        c1, c2 = st.columns(2)
        n_mc = int(c1.number_input("Monte Carlo samples", 3, 2000, 30, 1, key="igs_mc_n"))
        strategies_mc = c2.multiselect("Strategies", list(SCENARIOS.keys()), default=list(SCENARIOS.keys()), key="igs_mc_strategies")
        fixed_seed = st.checkbox("Fix S3 optimizer seed to isolate physical uncertainty", value=True, key="igs_mc_fixed_seed")
        eval_count = n_mc * max(len(strategies_mc), 1)
        st.caption(f"Estimated official solver evaluations: {eval_count:,}.")
        if st.button("Run paired Monte Carlo", type="primary", key="igs_run_mc"):
            try:
                if not strategies_mc:
                    raise ValueError("Select at least one strategy.")
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_paired_monte_carlo(runner, specs_all, strategies_mc, severity, climate, int(seed), n_mc, output_root, fixed_seed, callback), runner)
                bar.progress(1.0); status.caption("Monte Carlo analysis completed.")
                st.session_state["igs_mc_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_mc_paths", {})
        if paths:
            _preview_csv(paths.get("summary_csv"))
            _show_paths(paths, "igs_mc")

    with sub[2]:
        st.markdown("#### Morris global screening")
        c1, c2, c3 = st.columns(3)
        trajectories = int(c1.number_input("Morris trajectories", 2, 100, 10, 1, key="igs_morris_r"))
        levels = int(c2.number_input("Grid levels", 4, 20, 6, 2, key="igs_morris_p"))
        morris_kpi = c3.selectbox("Morris output KPI", STANDARD_KPIS, format_func=lambda x: KPI_LABELS[x], key="igs_morris_kpi")
        selected_names = st.multiselect("Parameters for Morris", [s.name for s in specs_all], default=[s.name for s in specs_all[:12]], format_func=lambda x: next(s.label for s in specs_all if s.name == x), key="igs_morris_params")
        specs = [s for s in specs_all if s.name in selected_names]
        st.caption(f"Estimated official solver evaluations: {trajectories * (len(specs) + 1):,}. Use 20–30 trajectories for final reporting after a diagnostic run.")
        if st.button("Run Morris screening", type="primary", key="igs_run_morris"):
            try:
                if not specs:
                    raise ValueError("Select at least one Morris parameter.")
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_morris(runner, specs, strategy, severity, climate, int(seed), trajectories, levels, morris_kpi, output_root, callback), runner)
                bar.progress(1.0); status.caption("Morris screening completed.")
                st.session_state["igs_morris_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_morris_paths", {})
        if paths:
            _preview_csv(paths.get("indices_csv"))
            _show_paths(paths, "igs_morris")

    with sub[3]:
        st.markdown("#### Sobol first-order and total-order sensitivity")
        c1, c2, c3 = st.columns(3)
        base_n = int(c1.number_input("Sobol base sample N", 4, 4096, 32, 4, key="igs_sobol_n"))
        bootstrap_reps = int(c2.number_input("Bootstrap repetitions", 0, 5000, 200, 50, key="igs_sobol_boot"))
        sobol_kpi = c3.selectbox("Sobol output KPI", STANDARD_KPIS, format_func=lambda x: KPI_LABELS[x], key="igs_sobol_kpi")
        selected_names = st.multiselect("Parameters for Sobol", [s.name for s in specs_all], default=[s.name for s in specs_all[:8]], format_func=lambda x: next(s.label for s in specs_all if s.name == x), key="igs_sobol_params")
        specs = [s for s in specs_all if s.name in selected_names]
        st.caption(f"Estimated official solver evaluations: {base_n * (len(specs) + 2):,}. Begin with N=32–64; use N=256–1024 for final results when computationally feasible.")
        if st.button("Run Sobol analysis", type="primary", key="igs_run_sobol"):
            try:
                if not specs:
                    raise ValueError("Select at least one Sobol parameter.")
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_sobol(runner, specs, strategy, severity, climate, int(seed), base_n, sobol_kpi, bootstrap_reps, output_root, callback), runner)
                bar.progress(1.0); status.caption("Sobol analysis completed.")
                st.session_state["igs_sobol_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_sobol_paths", {})
        if paths:
            _preview_csv(paths.get("indices_csv"))
            st.info("Do not clip negative or >1 finite-sample estimates. Increase N and check confidence intervals instead.")
            _show_paths(paths, "igs_sobol")

    with sub[4]:
        st.markdown("#### Benchmark stress cases")
        benchmark_strategies = st.multiselect("Benchmark strategies", list(SCENARIOS.keys()), default=list(SCENARIOS.keys()), key="igs_bench_strategies")
        if st.button("Run benchmark cases", type="primary", key="igs_run_benchmark"):
            try:
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_benchmarks(runner, {s.name: s.baseline for s in specs_all}, benchmark_strategies, severity, climate, int(seed), output_root, callback), runner)
                bar.progress(1.0); status.caption("Benchmark analysis completed.")
                st.session_state["igs_benchmark_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_benchmark_paths", {})
        if paths:
            _preview_csv(paths.get("results_csv"))
            _show_paths(paths, "igs_benchmark")

    with sub[5]:
        st.markdown("#### S3 control–maintenance–degradation-feedback ablation")
        if st.button("Run S3 ablation", type="primary", key="igs_run_ablation"):
            try:
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_ablation(runner, {s.name: s.baseline for s in specs_all}, severity, climate, int(seed), output_root, callback), runner)
                bar.progress(1.0); status.caption("Ablation analysis completed.")
                st.session_state["igs_ablation_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_ablation_paths", {})
        if paths:
            _preview_csv(paths.get("results_csv"))
            _show_paths(paths, "igs_ablation")

    with sub[6]:
        st.markdown("#### S3 optimizer population–iteration–seed robustness")
        c1, c2, c3 = st.columns(3)
        pop_text = c1.text_input("Populations", "12,18,24,30", key="igs_opt_pop")
        iter_text = c2.text_input("Iterations", "5,10,15,20", key="igs_opt_iter")
        seed_text = c3.text_input("Seeds", "41,42,43,44,45", key="igs_opt_seeds")
        try:
            pops = _parse_int_list(pop_text)
            iters = _parse_int_list(iter_text)
            seeds = _parse_int_list(seed_text, minimum=0)
            st.caption(f"Estimated official solver evaluations: {len(pops) * len(iters) * len(seeds):,}.")
        except Exception:
            pops, iters, seeds = [], [], []
        if st.button("Run optimizer robustness", type="primary", key="igs_run_optimizer"):
            try:
                if not pops or not iters or not seeds:
                    raise ValueError("Population, iteration, and seed lists must contain integers.")
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_optimizer_robustness(runner, {s.name: s.baseline for s in specs_all}, severity, climate, seeds, pops, iters, output_root, callback), runner)
                bar.progress(1.0); status.caption("Optimizer robustness completed.")
                st.session_state["igs_optimizer_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_optimizer_paths", {})
        if paths:
            _preview_csv(paths.get("summary_csv"))
            _show_paths(paths, "igs_optimizer")

    with sub[7]:
        st.markdown("#### Parameter-connectivity diagnostic")
        st.caption("Perturbs each enabled parameter to its upper bound and verifies that it reaches at least one official model KPI.")
        if st.button("Run connectivity test", type="primary", key="igs_run_connectivity"):
            try:
                runner = make_runner()
                bar, status, callback = _render_progress()
                paths = _attach_runner_cache_stats(run_connectivity_test(runner, specs_all, strategy, severity, climate, int(seed), output_root, callback), runner)
                bar.progress(1.0); status.caption("Connectivity test completed.")
                st.session_state["igs_connectivity_paths"] = paths
            except Exception as exc:
                st.exception(exc)
        paths = st.session_state.get("igs_connectivity_paths", {})
        if paths:
            _preview_csv(paths.get("results_csv"))
            _show_paths(paths, "igs_connectivity")

    with sub[8]:
        st.markdown("#### Separate extraction index")
        keys = [
            ("Local OAT", "igs_oat_paths"),
            ("Monte Carlo", "igs_mc_paths"),
            ("Morris", "igs_morris_paths"),
            ("Sobol", "igs_sobol_paths"),
            ("Benchmarks", "igs_benchmark_paths"),
            ("Ablation", "igs_ablation_paths"),
            ("Optimizer", "igs_optimizer_paths"),
            ("Connectivity", "igs_connectivity_paths"),
        ]
        index_rows = []
        for analysis, key in keys:
            paths = st.session_state.get(key, {})
            for result_name, value in paths.items():
                p = Path(str(value))
                if p.exists() and p.is_file():
                    index_rows.append({"analysis": analysis, "result": result_name, "file": p.name, "path": str(p), "size_kB": round(p.stat().st_size / 1024.0, 2)})
        if index_rows:
            index_df = pd.DataFrame(index_rows)
            st.dataframe(index_df, width="stretch")
            st.download_button("Download sensitivity result index", index_df.to_csv(index=False).encode("utf-8"), file_name="sensitivity_result_index.csv", mime="text/csv", key="igs_download_index")
            for analysis, key in keys:
                paths = st.session_state.get(key, {})
                if paths:
                    with st.expander(f"{analysis} downloads", expanded=False):
                        _show_paths(paths, f"igs_index_{_slug(analysis)}")
        else:
            st.info("Run one or more analyses; each table, figure, metadata file, and analysis ZIP will appear here separately.")
