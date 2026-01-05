## Overview

This project evaluates the robustness of classical visual object tracking algorithms under challenging visual conditions. Three OpenCV-based trackers (CSRT, KCF, and MedianFlow) are compared across multiple benchmark sequences under baseline, occlusion, and noise scenarios. Performance is evaluated using average IoU, success rate, and frames per second (FPS).

## Arborescence of the project

project/
├── data/OTB/ # OTB sequences (img/ + groundtruth_rect.txt)
├── trackers.py # Tracker initialization
├── degradation.py # Occlusion and noise simulation
├── metrics.py # IoU computation
├── run_experiment.py # Main experiment script
├── results/ # CSV result files
└── README.md

## Software requirements

-Python 3.8+
-OpenCV 4.5+ (with contrib modules)
-NumPy
-Pandas
(Command for installation in the requirements.txt)

## Run the program

# Navigate to the project directory

cd project

# To run the main experiment

python run_experiment.py

# To make analysis of average IoU and sucess rate

python analysis.py

# To see the plot of results

python plot.py

## Output

The following files are generated:

- baseline_results.csv
- occlusion_results.csv
- noise_results.csv
- fps_results.csv
- Summary tables used for analysis and reporting
