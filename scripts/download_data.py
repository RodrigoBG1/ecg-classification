"""
Instructions and validation for downloading the PTB-XL+ dataset from PhysioNet.

Run this script to see download instructions or to verify that the data
is already present at the expected location.
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_FILES = [
    "ptbxl_database.csv",
    "scp_statements.csv",
    "records100",
]

DOWNLOAD_URL = "https://physionet.org/files/ptb-xl-plus/1.0.1/"

INSTRUCTIONS = f"""
========================================================
  PTB-XL+ Dataset Download Instructions
========================================================

Dataset: PTB-XL+  v1.0.1
Source : PhysioNet (https://physionet.org/content/ptb-xl-plus/1.0.1/)
Size   : ~3.7 GB (compressed)

── Step 1: Register on PhysioNet ─────────────────────
  Create a free account at https://physionet.org/register/
  Then sign the data use agreement for PTB-XL+.

── Step 2: Download with wget ────────────────────────
  Run the following command (requires a PhysioNet account):

    wget -r -N -c -np \\
      --user=<YOUR_USERNAME> \\
      --ask-password \\
      {DOWNLOAD_URL}

  Or with curl:
    curl -u <YOUR_USERNAME> -O \\
      {DOWNLOAD_URL}ptbxl_database.csv
    curl -u <YOUR_USERNAME> -O \\
      {DOWNLOAD_URL}scp_statements.csv
    ... (then download records100/ recursively)

── Step 3: Place files ───────────────────────────────
  Extract / move so the layout looks like:

    data/raw/
    ├── ptbxl_database.csv
    ├── scp_statements.csv
    └── records100/
        ├── 00000/
        │   ├── 00001_lr.dat
        │   ├── 00001_lr.hea
        │   └── ...
        └── ...

── Alternative: WFDB download tool ──────────────────
  pip install wfdb
  python -c "import wfdb; wfdb.dl_database('ptb-xl-plus', './data/raw/')"

  Note: This may be slow for the full ~3.7 GB dataset.
========================================================
"""


def check_data(data_path: Path) -> bool:
    """Return True if all required files/directories are present."""
    all_ok = True
    print(f"\nChecking data at: {data_path.resolve()}\n")

    for name in REQUIRED_FILES:
        target = data_path / name
        if target.exists():
            size = (
                f"{target.stat().st_size / 1e6:.1f} MB"
                if target.is_file()
                else f"directory ({sum(1 for _ in target.rglob('*'))} items)"
            )
            print(f"  [OK]  {name}  ({size})")
        else:
            print(f"  [MISSING]  {name}")
            all_ok = False

    return all_ok


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PTB-XL+ download helper")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("data/raw/"),
        help="Where to look for the dataset (default: data/raw/)",
    )
    args = parser.parse_args()

    data_ready = check_data(args.data_path)

    if data_ready:
        print("\nAll required files found. Dataset is ready to use.\n")
    else:
        print(INSTRUCTIONS)
        print(
            "After downloading, re-run this script to verify:\n"
            f"  python scripts/download_data.py --data_path {args.data_path}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
