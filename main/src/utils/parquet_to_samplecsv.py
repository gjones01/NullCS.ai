from __future__ import annotations

from pathlib import Path
import zipfile
import io

# =========================
# CONFIG (edit these only)
# =========================

# Option A: direct parquet file path
PARQUET_PATH = Path(r"C:\NullCS\some_folder\kills.parquet")

# Option B: parquet inside a zip (set ZIP_PATH and INNER_PARQUET_NAME)
# ZIP_PATH = Path(r"C:\NullCS\parsed_zips\Normal002.zip")
# INNER_PARQUET_NAME = "kills.parquet"

OUT_CSV_PATH = Path(r"C:\NullCS\main\data\samples\kills_sample.csv")

# How to sample:
N_ROWS = 10_000            # number of rows to output
METHOD = "random"          # "random" or "head"
SEED = 42                  # used only for random sampling

# Optional: only keep certain columns (leave empty list to keep all)
COLUMNS: list[str] = []    # e.g. ["tick", "round_num", "attacker_steamid", "victim_steamid"]

# CSV formatting
INCLUDE_INDEX = False
ENCODING = "utf-8"


# =========================
# IMPLEMENTATION
# =========================

def load_parquet_as_pandas_from_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    # Prefer Polars for speed; fall back to pandas
    try:
        import polars as pl
        df = pl.read_parquet(str(path))
        return df.to_pandas()
    except Exception:
        import pandas as pd
        return pd.read_parquet(path)


def load_parquet_as_pandas_from_zip(zip_path: Path, inner_name: str):
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        if inner_name not in z.namelist():
            raise FileNotFoundError(
                f"'{inner_name}' not found in zip. موجود: {z.namelist()[:25]} ..."
            )
        data = z.read(inner_name)

    # Read parquet from bytes
    try:
        import polars as pl
        df = pl.read_parquet(io.BytesIO(data))
        return df.to_pandas()
    except Exception:
        import pandas as pd
        import pyarrow.parquet as pq
        table = pq.read_table(io.BytesIO(data))
        return table.to_pandas()


def sample_df(df):
    if COLUMNS:
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in parquet: {missing}")
        df = df[COLUMNS]

    if N_ROWS <= 0:
        return df

    if METHOD == "head":
        return df.head(N_ROWS)

    if METHOD == "random":
        n = min(N_ROWS, len(df))
        return df.sample(n=n, random_state=SEED)

    raise ValueError("METHOD must be 'random' or 'head'")


def main():
    OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Choose source:
    use_zip = "ZIP_PATH" in globals() and "INNER_PARQUET_NAME" in globals()

    if use_zip:
        zip_path = globals()["ZIP_PATH"]
        inner = globals()["INNER_PARQUET_NAME"]
        print(f"[LOAD] zip={zip_path} inner={inner}")
        df = load_parquet_as_pandas_from_zip(zip_path, inner)
    else:
        print(f"[LOAD] parquet={PARQUET_PATH}")
        df = load_parquet_as_pandas_from_file(PARQUET_PATH)

    print(f"[INFO] rows={len(df):,} cols={len(df.columns):,}")

    out = sample_df(df)
    out.to_csv(OUT_CSV_PATH, index=INCLUDE_INDEX, encoding=ENCODING)

    print(f"[OK] wrote sample CSV: {OUT_CSV_PATH} (rows={len(out):,})")


if __name__ == "__main__":
    main()
