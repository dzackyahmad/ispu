"""
Perekam ISPU KLHK -> data/ispu_log.csv
Dijalankan tiap 30 menit oleh GitHub Actions.
Menyimpan snapshot getStations, dikonversi ke UTC + ug/m3 + cell_id grid 5 km.
"""

import io
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

URL = "https://ispu.kemenlh.go.id/apimobile/v1/getStations"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://ispu.kemenlh.go.id/webv5/",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
}

OUT = "data/ispu_log.csv"

KOLOM = [
    "id_stasiun", "nama", "kota", "provinsi", "p3e",
    "waktu", "time_z", "lat", "lon", "tipe_text", "is_maintenance",
    "t_pm25", "a_pm25", "c_pm25",
    "t_pm10", "a_pm10", "t_so2", "t_co", "t_o3", "t_no2",
]

OFFSET_JAM = {"WIB": 7, "WITA": 8, "WIT": 9}

# Grid 5x5 km (1/24 derajat) - jangkar dari idn_pop_2026_5x5_extracted.csv
H = 1.0 / 24.0
LON0 = 95.04583223315004
LAT0 = -10.97916628675

# Breakpoint ISPU PM2.5, Permen LHK 14/2020: (ISPU_bawah, ISPU_atas, konsentrasi_bawah, konsentrasi_atas)
BREAKPOINT = [
    (0, 50, 0.0, 15.5),
    (51, 100, 15.6, 55.4),
    (101, 200, 55.5, 150.4),
    (201, 300, 150.5, 250.4),
    (301, 500, 250.5, 500.0),
]


def ispu_ke_ugm3(indeks):
    """Balik indeks ISPU PM2.5 menjadi konsentrasi ug/m3."""
    if indeks is None or (isinstance(indeks, float) and np.isnan(indeks)):
        return np.nan
    try:
        x = float(indeks)
    except (TypeError, ValueError):
        return np.nan
    if x < 0:
        return np.nan
    if x > 500:
        x = 500.0
    for lo, hi, clo, chi in BREAKPOINT:
        if lo <= x <= hi:
            return clo + (x - lo) * (chi - clo) / (hi - lo)
    return np.nan


def ke_sel(lon, lat):
    """Koordinat -> indeks sel grid (i, j). Aman terhadap nilai kosong."""
    lon = pd.to_numeric(pd.Series(lon).reset_index(drop=True), errors="coerce")
    lat = pd.to_numeric(pd.Series(lat).reset_index(drop=True), errors="coerce")
    i = ((lon - LON0) / H).round().astype("Int64")
    j = ((lat - LAT0) / H).round().astype("Int64")
    return i, j


def angka(seri):
    return pd.to_numeric(seri, errors="coerce")


def ambil():
    r = requests.get(URL, headers=HEADERS, timeout=120)
    r.raise_for_status()
    rows = r.json()["rows"]
    df = pd.DataFrame(rows)

    for k in KOLOM:
        if k not in df.columns:
            df[k] = np.nan
    df = df[KOLOM].copy()

    # ---- waktu: lokal -> UTC ----
    lokal = pd.to_datetime(df["waktu"], errors="coerce")
    jam = df["time_z"].map(OFFSET_JAM)
    df["waktu_utc"] = lokal - pd.to_timedelta(jam, unit="h")

    # ---- numerik ----
    for k in ["lat", "lon", "t_pm25", "a_pm25", "t_pm10", "a_pm10",
              "t_so2", "t_co", "t_o3", "t_no2", "c_pm25", "is_maintenance"]:
        df[k] = angka(df[k])

    # ---- label PM2.5 dalam ug/m3 ----
    langsung = df["a_pm25"].where(df["a_pm25"] > 0)
    konversi = df["t_pm25"].map(ispu_ke_ugm3)
    df["pm25_ugm3"] = langsung.fillna(konversi)
    df["sumber_label"] = np.where(
        df["a_pm25"] > 0, "ISPU-langsung",
        np.where(df["t_pm25"].notna(), "ISPU-konversi", "kosong"))

    # ---- sel grid ----
    df["i"], df["j"] = ke_sel(df["lon"], df["lat"])
    df["cell_id"] = df["i"] * 100000 + df["j"]

    df["diambil_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    df = df.dropna(subset=["waktu_utc"])
    return df


def main():
    baru = ambil()
    print(f"snapshot: {len(baru)} baris")

    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT):
        lama = pd.read_csv(OUT, parse_dates=["waktu_utc"])
        gab = pd.concat([lama, baru], ignore_index=True)
    else:
        gab = baru

    sebelum = len(gab)
    gab = gab.drop_duplicates(subset=["id_stasiun", "waktu_utc"], keep="first")
    gab = gab.sort_values(["waktu_utc", "id_stasiun"]).reset_index(drop=True)
    gab.to_csv(OUT, index=False)

    print(f"duplikat dibuang : {sebelum - len(gab)}")
    print(f"total baris      : {len(gab):,}")
    print(f"jam unik         : {gab['waktu_utc'].nunique()}")
    print(f"sel unik         : {gab['cell_id'].nunique()}")
    print(f"rentang          : {gab['waktu_utc'].min()}  s/d  {gab['waktu_utc'].max()}")
    print(gab["sumber_label"].value_counts().to_string())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"GAGAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)