import zipfile, io
import polars as pl
from pathlib import Path

zip_path = Path(r"C:\NullCS\parsed_zips\Normal001.zip")

with zipfile.ZipFile(zip_path, "r") as z:
    for name in ["ticks.parquet", "kills.parquet", "shots.parquet", "damages.parquet", "infernos.parquet", "smokes.parquet", "grenades.parquet"]:
        print("\n====", name, "====")
        if name not in z.namelist():
            print("NOT FOUND")
            continue
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        print("rows:", df.height, "cols:", len(df.columns))
        print(df.columns)
