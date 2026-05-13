"""
download_data.py
----------------
Downloads the PhysioNet EEG Motor Movement/Imagery Dataset for all
103 usable subjects using MNE-Python's built-in fetcher.

Compatible with Python 3.9+ and all MNE versions (1.x).

Usage:
    python3 download_data.py
    python3 download_data.py --subjects 1 2 3
"""

import argparse
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ALL_SUBJECTS, ALL_MI_RUNS, DATA_DIR


def download_subject(subject_id, path):
    """
    Download all MI runs for one subject.
    Tries both 'subjects' (MNE >= 1.6) and 'subject' (older MNE) signatures.
    Returns list of local file paths, or empty list on failure.
    """
    from mne.datasets import eegbci
    import inspect

    sig = inspect.signature(eegbci.load_data)
    first_param = list(sig.parameters.keys())[0]

    try:
        if first_param == 'subjects':
            files = eegbci.load_data(
                subjects=subject_id,
                runs=ALL_MI_RUNS,
                path=path,
                verbose=False,
            )
        else:
            # Older MNE: singular 'subject'
            files = eegbci.load_data(
                subject=subject_id,
                runs=ALL_MI_RUNS,
                path=path,
                verbose=False,
            )
        return list(files)
    except Exception as e:
        tqdm.write(f"  [WARN] Subject {subject_id:03d}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet EEGMMIDB")
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=None,
        help="Which subjects to download (default: all 103 usable subjects)"
    )
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else ALL_SUBJECTS

    print(f"Downloading PhysioNet EEGMMIDB for {len(subjects)} subjects ...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Runs per subject: {ALL_MI_RUNS}  ({len(ALL_MI_RUNS)} files each)")
    print()

    total_files = 0
    failed = []

    for subj in tqdm(subjects, desc="Subjects"):
        files = download_subject(subj, DATA_DIR)
        if files:
            total_files += len(files)
        else:
            failed.append(subj)

    print(f"\nDone. Downloaded {total_files} EDF files total.")
    if failed:
        print(f"Failed subjects: {failed}")
    else:
        print("All subjects downloaded successfully.")


if __name__ == "__main__":
    main()
