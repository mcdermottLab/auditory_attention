import os
import subprocess
from pathlib import Path
import sys

# Add the src directory to the Python path for this script
project_root = Path(__file__).resolve().parents[2]  # Go up two levels to the project root
src_dir = project_root 
sys.path.append(str(src_dir))

# Directory containing the scripts
scripts_dir = Path(__file__).parent

# Output directory for all figures
figure_out_dir = scripts_dir / "all_figures_output"
figure_out_dir.mkdir(parents=True, exist_ok=True)

# List all Python scripts in the directory
script_files = [f for f in os.listdir(scripts_dir) if f.endswith(".py") and f != "run_all_figures.py"]


script_files = [
    "plot_experiment_1_and_sup_fig_1.py",
    "plot_experiment_2.py",
    "plot_experiment_3.py",
    "plot_experiment_4_and_sup_fig_7.py",
    "plot_experiment_5.py",
    "plot_figure_4a_and_supplementary_figures_2_and_3.py",
    "plot_supplementary_figure_4.py",
    "plot_experiment_6.py",
    "interaction_test_for_experiment_6.py",
    "plot_experiment_7.py",
    "plot_figure_5_and_sup_figure_5.py",
    "plot_figure_6.py",
]

# Execute each script sequentially
for script_file in script_files:
    script_path = os.path.join(scripts_dir, script_file)
    print(f"Executing {script_path}...")

    # Run the script and pass the figure_out_dir as an argument
    result = subprocess.run(
        ["python3", script_path, str(figure_out_dir)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src_dir)}  # Add src to PYTHONPATH
    )

    # Print the output of the script
    print(result.stdout)
    if result.stderr:
        if "Error" in result.stderr:
            print(f"Error in {script_file}:\n{result.stderr}")
        else:
            print(f"Output from {script_file}:\n{result.stderr}")