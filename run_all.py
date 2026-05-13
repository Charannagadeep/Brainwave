"""
run_all.py
----------
Runs the complete BrainWave pipeline end-to-end:

  1. Download PhysioNet data (all 103 subjects)
  2. Preprocess all subjects
  3. CSP+SVM and CSP+Random Forest baselines
  4. Train Spatial CNN Only (subject-dependent)
  5. Train CNN+BiLSTM (subject-dependent)
  6. Train BrainWave L=4 (subject-dependent)
  7. Train BrainWave L=1,2,6 (depth ablation)
  8. Train BrainWave (subject-independent, 5-fold)
  9. Evaluate and print summary table
  10. Generate visualizations

Total time on a Kaggle T4 GPU: ~4-6 hours for all 103 subjects.
For a quick test, pass --subjects 1 2 3 to run on 3 subjects only.

Usage:
    python run_all.py
    python run_all.py --subjects 1 2 3     # quick test
    python run_all.py --skip_download      # if data already downloaded
    python run_all.py --skip_preprocess    # if already preprocessed
"""

import argparse
import subprocess
import sys
import os


def run(cmd: list[str], desc: str = ""):
    """Run a subprocess command and print output."""
    if desc:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable] + cmd,
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {' '.join(cmd)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run full BrainWave pipeline")
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Limit to specific subjects (e.g. --subjects 1 2 3)")
    parser.add_argument("--skip_download",    action="store_true")
    parser.add_argument("--skip_preprocess",  action="store_true")
    parser.add_argument("--skip_baselines",   action="store_true")
    parser.add_argument("--skip_train",       action="store_true")
    parser.add_argument("--skip_visualize",   action="store_true")
    args = parser.parse_args()

    # Build the subject flag to pass to sub-scripts
    subj_flag = []
    if args.subjects:
        subj_flag = ["--subjects"] + [str(s) for s in args.subjects]

    # ── Step 1: Download ──────────────────────────────────────────────────────
    if not args.skip_download:
        run(["download_data.py"] + subj_flag, "Step 1: Downloading PhysioNet data")
    else:
        print("\n[SKIP] Download")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    if not args.skip_preprocess:
        run(["preprocess.py"] + subj_flag, "Step 2: Preprocessing EEG epochs")
    else:
        print("\n[SKIP] Preprocessing")

    # ── Step 3: Baselines ─────────────────────────────────────────────────────
    if not args.skip_baselines:
        run(
            ["baselines.py"] + subj_flag,
            "Step 3: CSP + SVM and CSP + Random Forest baselines"
        )
    else:
        print("\n[SKIP] Baselines")

    # ── Step 4–8: Train models ────────────────────────────────────────────────
    if not args.skip_train:

        run(
            ["train.py", "--mode", "subj_dep", "--model", "cnn_only"] + subj_flag,
            "Step 4: Training Spatial CNN Only (subject-dependent)"
        )

        run(
            ["train.py", "--mode", "subj_dep", "--model", "cnn_bilstm"] + subj_flag,
            "Step 5: Training CNN + BiLSTM (subject-dependent)"
        )

        run(
            ["train.py", "--mode", "subj_dep", "--model", "brainwave",
             "--layers", "4"] + subj_flag,
            "Step 6: Training BrainWave L=4 (subject-dependent)"
        )

        # Depth ablation (L=1, 2, 6 — L=4 already done above)
        for L in [1, 2, 6]:
            run(
                ["train.py", "--mode", "subj_dep", "--model", "brainwave",
                 "--layers", str(L)] + subj_flag,
                f"Step 7 (ablation): Training BrainWave L={L}"
            )

        run(
            ["train.py", "--mode", "subj_ind", "--model", "brainwave",
             "--folds", "5"],
            "Step 8: Training BrainWave subject-independent (5-fold)"
        )

    else:
        print("\n[SKIP] Training")

    # ── Step 9: Evaluate ──────────────────────────────────────────────────────
    run(["evaluate.py"], "Step 9: Collecting results and printing summary")

    # ── Step 10: Visualize ────────────────────────────────────────────────────
    if not args.skip_visualize:
        run(["visualize.py", "--all"], "Step 10: Generating attention visualizations")
    else:
        print("\n[SKIP] Visualization")

    print("\n" + "="*60)
    print("  Pipeline complete! Check results/ for CSVs and figures.")
    print("="*60)


if __name__ == "__main__":
    main()
