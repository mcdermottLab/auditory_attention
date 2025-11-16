import argparse
import os
import subprocess
from pathlib import Path
import sys

from final_figures_paths import DATA_ROOT, PROJECT_ROOT

# Add the src directory to the Python path for this script
src_dir = PROJECT_ROOT
sys.path.append(str(src_dir))
python_executable = sys.executable

parser = argparse.ArgumentParser(description="Run all Final Figures scripts")
parser.add_argument("--dry-run", action="store_true", help="Propagate dry-run flag to individual scripts")
args = parser.parse_args()

# Directory containing the scripts
scripts_dir = Path(__file__).parent

# Output directory for all figures
figure_out_dir = scripts_dir / "all_figures_output"
figure_out_dir.mkdir(parents=True, exist_ok=True)

# List all Python scripts in the directory
script_files = [f for f in os.listdir(scripts_dir) if f.endswith(".py") and f != "run_all_figure_gen.py"]


script_files = [
    "plot_experiment_1_and_sup_fig_1.py",
    "plot_experiment_2.py",
    "plot_experiment_3.py",
    "plot_experiment_4.py",
    "plot_experiment_5.py",
    "plot_experiment_6.py",
    "plot_experiment_7.py",
    "interaction_test_for_experiment_6.py",
    "individual_participants_fig_2.py",
    "plot_figure_4a_and_supplementary_figures_x_and_x.py",
    "plot_figure_5_b-c.py",
    "plot_figure_6_a-c.py",
    "plot_figure_6_d.py",
    "plot_supplementary_figure_3.py",
    "plot_supplementary_figure_7_abd.py",
    "plot_supplementary_figure_9.py",
    "plot_sup_figure_10.py",
]

# Execute each script sequentially
for script_file in script_files:
    script_path = os.path.join(scripts_dir, script_file)
    print(f"Executing {script_path}...")

    # Run the script from the project root so relative paths (e.g., participant_data)
    # match the original notebooks' expectations. Pass the output directory as an absolute path.
    cmd = [python_executable, script_path, "--fig-dir", str(figure_out_dir)]
    if args.dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=str(PROJECT_ROOT),
        env={
            **os.environ,
            "PYTHONPATH": str(src_dir),  # Add src to PYTHONPATH
            "FINAL_FIGURES_DATA_ROOT": str(DATA_ROOT),
        }
    )

    # Print the output of the script
    print(result.stdout)
    if result.stderr:
        if "Error" in result.stderr:
            print(f"Error in {script_file}:\n{result.stderr}")
        else:
            print(f"Output from {script_file}:\n{result.stderr}")

