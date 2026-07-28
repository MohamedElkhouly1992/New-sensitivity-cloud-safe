#!/usr/bin/env python3
"""Calibrate bounded hidden operational-state factors from DesignBuilder and ROM daily outputs.

Usage:
    python calibrate_operational_state_factors.py \
        --designbuilder_csv "ALL DATA - Design builder Data.csv" \
        --solver_csv baseline_no_degradation_daily.csv \
        --out_csv operational_state_factors.csv

The script keeps the reduced-order model physics intact. It tunes regime-level
component multipliers only, using a compact genetic algorithm to reduce daily
CVRMSE/MAPE while controlling monthly and annual bias. Use training years for
calibration, then validate on a holdout year or leave-one-year-out split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple
import numpy as np
import pandas as pd

STATE_LABELS = (
    "R0_low_operation",
    "R1_heating_dominated",
    "R2_mixed_shoulder",
    "R3_early_cooling",
    "R4_peak_cooling",
    "R5_late_cooling_academic",
    "R6_extreme_high_load",
)


def _find_col(df: pd.DataFrame, candidates: Iterable[str], contains: Iterable[str] = ()) -> str:
    lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower:
            return lower[key]
    for c in df.columns:
        lc = str(c).strip().lower()
        if all(s.lower() in lc for s in contains):
            return c
    raise KeyError(f"Could not find a column matching {list(candidates)} / contains={list(contains)}")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def load_daily_pair(db_path: Path, sol_path: Path, start_year: int = 2020) -> pd.DataFrame:
    db = _read_csv_flexible(db_path)
    sol = _read_csv_flexible(sol_path)
    db_date = _find_col(db, ["Date/Time", "date", "Date", "timestamp"], contains=["date"])
    try:
        db_energy = _find_col(db, ["Total Design builder", "DesignBuilder", "designbuilder_kwh", "DB energy", "Total Energy"], contains=["design"])
    except KeyError:
        db_energy = _find_col(db, ["Total Energy", "total_energy", "Energy"], contains=["total", "energy"])
    db2 = pd.DataFrame({"date": pd.to_datetime(db[db_date], errors="coerce"), "E_db": pd.to_numeric(db[db_energy], errors="coerce")})
    # Drop DesignBuilder unit/header rows and any non-numeric metadata rows. If the
    # Total Energy column is absent or empty in a specific export, use common energy
    # components as fallback.
    if db2["E_db"].notna().sum() == 0:
        comp_candidates = [c for c in ["Cooling (Electricity)", "System Fans", "System Pumps", "Auxiliary Energy", "Heating (Gas)"] if c in db.columns]
        if comp_candidates:
            db2["E_db"] = sum(pd.to_numeric(db[c], errors="coerce").fillna(0.0) for c in comp_candidates)
    db2 = db2.dropna(subset=["date", "E_db"])
    db2["date_key"] = db2["date"].dt.date
    db_daily = db2.groupby("date_key", as_index=False).agg(E_db=("E_db", "sum"))
    db_daily = db_daily.rename(columns={"date_key": "date"})
    db_daily["date"] = pd.to_datetime(db_daily["date"])

    # Solver date handling: use existing date if available, otherwise construct it
    # from year/day_of_year. Solver year can be either absolute (2020) or 1-based.
    sol2 = sol.copy()
    try:
        sol_date = _find_col(sol2, ["Date/Time", "date", "Date", "timestamp"], contains=["date"])
        sol2["date"] = pd.to_datetime(sol2[sol_date], errors="coerce")
    except KeyError:
        ycol = _find_col(sol2, ["year", "Year"], contains=["year"])
        dcol = _find_col(sol2, ["day_of_year", "doy", "day"], contains=["day"])
        y_raw = pd.to_numeric(sol2[ycol], errors="coerce").fillna(1).astype(int)
        # Convert 1..N simulation years to absolute years.
        y_abs = np.where(y_raw < 1900, start_year + y_raw - 1, y_raw)
        doy = pd.to_numeric(sol2[dcol], errors="coerce").fillna(1).clip(lower=1).astype(int)
        sol2["date"] = pd.to_datetime(y_abs.astype(str) + "-01-01", errors="coerce") + pd.to_timedelta(doy - 1, unit="D")
    sol_energy = _find_col(sol2, ["energy_kwh_day", "energy_kwh_period", "Model energy", "model_kwh"], contains=["energy"])
    sol2["E_model"] = pd.to_numeric(sol2[sol_energy], errors="coerce")
    cols_mean = []
    for c in ["T_amb_C", "GHI_mean_Wm2", "RH_mean_pct", "occ", "Q_cool_kw", "Q_heat_kw"]:
        if c in sol2.columns:
            sol2[c] = pd.to_numeric(sol2[c], errors="coerce")
            cols_mean.append(c)
    agg = {"E_model": ("E_model", "sum")}
    for c in cols_mean:
        agg[c] = (c, "mean")
    for c in ["cooling_hvac_kwh_period", "heating_hvac_kwh_period", "fan_kwh_period", "pump_kwh_period", "auxiliary_kwh_period"]:
        if c in sol2.columns:
            sol2[c] = pd.to_numeric(sol2[c], errors="coerce")
            agg[c] = (c, "sum")
    sol2 = sol2.dropna(subset=["date", "E_model"])
    sol2["date_key"] = sol2["date"].dt.date
    sol_daily = sol2.groupby("date_key", as_index=False).agg(**agg)
    sol_daily = sol_daily.rename(columns={"date_key": "date"})
    sol_daily["date"] = pd.to_datetime(sol_daily["date"])

    out = pd.merge(db_daily, sol_daily, on="date", how="inner")
    out = out.dropna(subset=["E_db", "E_model"]).sort_values("date").reset_index(drop=True)
    if len(out) == 0:
        raise ValueError("No overlapping daily records found.")
    out["month"] = out["date"].dt.month
    out["doy"] = out["date"].dt.dayofyear.clip(upper=365)
    return out

def infer_state(row: pd.Series) -> int:
    m = int(row.get("month", 1))
    t = float(row.get("T_amb_C", np.nan))
    occ = float(row.get("occ", 0.5)) if pd.notna(row.get("occ", np.nan)) else 0.5
    qc = max(float(row.get("Q_cool_kw", 0.0)) if pd.notna(row.get("Q_cool_kw", np.nan)) else 0.0, 0.0)
    qh = max(float(row.get("Q_heat_kw", 0.0)) if pd.notna(row.get("Q_heat_kw", np.nan)) else 0.0, 0.0)
    qt = qc + qh
    cs = qc / max(qt, 1e-9)
    hs = qh / max(qt, 1e-9)
    if occ < 0.18 and qt <= 1e-6:
        return 0
    if pd.notna(t) and t >= 35.0:
        return 6
    if (m in (1, 2, 3, 12) and hs >= 0.45) or (pd.notna(t) and t <= 18.0):
        return 1
    if (cs > 0.20 and hs > 0.20) or m in (3, 4, 10, 11):
        return 2
    if m in (5, 6):
        return 3
    if m in (7, 8):
        return 4
    if m in (9, 10):
        return 5
    return 4 if cs >= hs else 1


def metrics(y: np.ndarray, yp: np.ndarray, dates: pd.Series) -> Dict[str, float]:
    e = yp - y
    mape = np.mean(np.abs(e) / np.maximum(np.abs(y), 1e-9)) * 100
    rmse = float(np.sqrt(np.mean(e ** 2)))
    cvrmse = rmse / max(np.mean(y), 1e-9) * 100
    nmbe = np.sum(e) / max(np.sum(y), 1e-9) * 100
    tmp = pd.DataFrame({"date": dates, "e": e, "db": y})
    tmp["month_id"] = tmp["date"].dt.to_period("M").astype(str)
    monthly = tmp.groupby("month_id").agg(e=("e", "sum"), db=("db", "sum"))
    monthly_abs_nmbe = np.mean(np.abs(monthly["e"] / np.maximum(monthly["db"], 1e-9))) * 100
    return {"MAPE": float(mape), "CVRMSE": float(cvrmse), "NMBE": float(nmbe), "monthly_abs_NMBE": float(monthly_abs_nmbe)}


def objective(df: pd.DataFrame, factors: np.ndarray, weights=(0.40, 0.25, 0.25, 0.10)) -> float:
    f = factors[df["state_index"].values]
    yp = df["E_model"].values * f
    m = metrics(df["E_db"].values, yp, df["date"])
    return weights[0] * m["CVRMSE"] + weights[1] * m["MAPE"] + weights[2] * m["monthly_abs_NMBE"] + weights[3] * abs(m["NMBE"])


def genetic_calibrate(df: pd.DataFrame, seed: int = 42, population: int = 80, generations: int = 120, lo: float = 0.60, hi: float = 1.45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.ones(7)
    for i in range(7):
        g = df[df["state_index"] == i]
        if len(g) > 0 and g["E_model"].sum() > 0:
            base[i] = np.clip(g["E_db"].sum() / g["E_model"].sum(), lo, hi)
    pop = np.vstack([base, np.ones(7), rng.uniform(lo, hi, size=(population - 2, 7))])
    best = base.copy()
    best_j = objective(df, best)
    for gen in range(generations):
        scores = np.array([objective(df, x) for x in pop])
        order = np.argsort(scores)
        pop = pop[order]
        if scores[order[0]] < best_j:
            best_j = float(scores[order[0]])
            best = pop[0].copy()
        elites = pop[: max(6, population // 8)]
        children = [best, np.ones(7), base]
        while len(children) < population:
            a, b = elites[rng.integers(0, len(elites), size=2)]
            mask = rng.random(7) < 0.5
            child = np.where(mask, a, b)
            sigma = max(0.03, 0.18 * (1 - gen / max(generations, 1)))
            child = child + rng.normal(0.0, sigma, size=7)
            children.append(np.clip(child, lo, hi))
        pop = np.array(children[:population])
    return np.clip(best, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designbuilder_csv", required=True)
    ap.add_argument("--solver_csv", required=True)
    ap.add_argument("--out_csv", default="operational_state_factors.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--population", type=int, default=80)
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--start_year", type=int, default=2020)
    args = ap.parse_args()
    df = load_daily_pair(Path(args.designbuilder_csv), Path(args.solver_csv), start_year=args.start_year)
    df["state_index"] = df.apply(infer_state, axis=1)
    before = metrics(df["E_db"].values, df["E_model"].values, df["date"])
    factors = genetic_calibrate(df, seed=args.seed, population=args.population, generations=args.generations)
    after_pred = df["E_model"].values * factors[df["state_index"].values]
    after = metrics(df["E_db"].values, after_pred, df["date"])
    rows = []
    for i, label in enumerate(STATE_LABELS):
        rows.append({
            "state_index": i,
            "state_label": label,
            "cooling_factor": factors[i],
            "heating_factor": factors[i],
            "fan_factor": factors[i],
            "pump_factor": factors[i],
            "aux_factor": factors[i],
            "n_days_in_training": int((df["state_index"] == i).sum()),
            "calibration_note": "GA factor replicated to components unless component-specific DesignBuilder data are supplied",
        })
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print("Before:", before)
    print("After:", after)

if __name__ == "__main__":
    main()
