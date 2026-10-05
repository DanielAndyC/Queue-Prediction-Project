"""
Run the full CYM two-stage queue pipeline:
  simulation → features → data_analysis → model → analysis → plots
"""
import subprocess
import sys
import os

STEPS = [
    ("Running simulation (two-stage)", "src/simulation.py"),
    ("Building feature table",         "src/features.py"),
    ("Exploring data",                 "src/data_analysis.py"),
    ("Training and evaluating models", "src/model.py"),
    ("Running cross-correlation and sensitivity", "src/analysis.py"),
    ("Generating plots",               "src/visualise.py"),
]


def run_step(label, script):
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    if subprocess.run([sys.executable, script]).returncode != 0:
        print(f"  FAILED: {script}")
        sys.exit(1)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for label, script in STEPS:
        run_step(label, script)
    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()