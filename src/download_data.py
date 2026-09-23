"""Download the MIT-BIH Arrhythmia Database and the MIT-BIH Noise Stress Test Database from PhysioNet.

Usage:
    python src/download_data.py            # both databases into data/raw/
    python src/download_data.py --force    # re-download even if files exist
"""
import argparse

import wfdb

from config import load_config, resolve

DATABASES = {
    "mitdb": "mitdb_dir",  # https://physionet.org/content/mitdb/1.0.0/
    "nstdb": "nstdb_dir",  # https://physionet.org/content/nstdb/1.0.0/
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    for db, key in DATABASES.items():
        out = resolve(cfg["paths"][key])
        if any(out.glob("*.hea")) and not args.force:
            print(f"[skip] {db}: already present in {out}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"[download] {db} -> {out}")
        wfdb.dl_database(db, dl_dir=str(out), overwrite=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
