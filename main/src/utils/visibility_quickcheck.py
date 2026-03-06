from pathlib import Path
import sys
import zipfile, io
import polars as pl

# Add: C:\NullCS\main\src to sys.path
SRC_DIR = Path(r"C:\NullCS\main\src")
sys.path.insert(0, str(SRC_DIR))

from utils.visibility_awpy import map_name_from_zip, tri_path_for_map, is_visible, Point3

ZIP_PATH = Path(r"C:\NullCS\parsed_zips\Normal001.zip")

def main():
    map_name = map_name_from_zip(ZIP_PATH)
    tri_path = tri_path_for_map(map_name)

    print("[INFO] map:", map_name)
    print("[OK] tri:", tri_path)

  
    # read first kill from the zip
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        kills = pl.read_parquet(io.BytesIO(z.read("kills.parquet")))

    row = kills.row(0, named=True)

    p1 = Point3(row["attacker_X"], row["attacker_Y"], row["attacker_Z"] + 64)  # eye height
    p2 = Point3(row["victim_X"], row["victim_Y"], row["victim_Z"] + 56)        # chest height

    print("[DEBUG] kill tick:", row["tick"])
    print("[DEBUG] attacker:", row["attacker_name"], row["attacker_steamid"])
    print("[DEBUG] victim  :", row["victim_name"], row["victim_steamid"])
    print("[TEST] LOS at kill positions:", is_visible(map_name, p1, p2))


if __name__ == "__main__":
    main()
