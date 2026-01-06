# ------------------------------------------------------------
# This script performs post-processing analysis of tracking
# results. It computes average IoU, success rates at three different
# thresholds, and average FPS for each tracker and experiment.
# ------------------------------------------------------------

import pandas as pd

# ===================== BASELINE RESULTS ANALYSIS =====================
print("###################### Baseline Experiment Results Summary ############################")

# Load baseline IoU results
df = pd.read_csv('results/baseline_results.csv')

# Compute average IoU per tracker
avg_iou = df.groupby(["Tracker"]).agg({"IoU": "mean"}).reset_index()

# Define success based on IoU thresholds (0.5, 0.4, and 0.3)
df["Success Rate 50"] = (df["IoU"] > 0.5).astype(int)
df["Success Rate 40"] = (df["IoU"] > 0.4).astype(int)
df["Success Rate 30"] = (df["IoU"] > 0.3).astype(int)

# Compute success rate percentages per tracker
success_rate_df = (
    df.groupby("Tracker")
    .agg({"Success Rate 50": "mean", "Success Rate 40": "mean", "Success Rate 30": "mean"})
    .reset_index()
)

success_rate_df["Success Rate 50"] = success_rate_df["Success Rate 50"] * 100
success_rate_df["Success Rate 40"] = success_rate_df["Success Rate 40"] * 100
success_rate_df["Success Rate 30"] = success_rate_df["Success Rate 30"] * 100

# Combine metrics into a summary table
summary = pd.DataFrame({
    "Tracker": avg_iou["Tracker"],
    "Average IoU": avg_iou["IoU"],
    "Success Rate 50 (%)": success_rate_df["Success Rate 50"],
    "Success Rate 40 (%)": success_rate_df["Success Rate 40"],
    "Success Rate 30 (%)": success_rate_df["Success Rate 30"]
})

# Display summary statistics
print(summary)
print("Average IoU:", summary["Average IoU"].mean())
print("Success Rate 50 (%):", summary["Success Rate 50 (%)"].mean())
print("Success Rate 40 (%):", summary["Success Rate 40 (%)"].mean())
print("Success Rate 30 (%):", summary["Success Rate 30 (%)"].mean())

# Save baseline summary to CSV
summary.to_csv("results/summary_baseline.csv", index=False)


# ===================== OCCLUSION RESULTS ANALYSIS =====================
print("\n###################### Occlusion Experiment Results Summary ############################")

# Load occlusion experiment results
df_occ = pd.read_csv('results/occlusion_results.csv')

# Compute average IoU per tracker under occlusion
df_occ = df_occ.groupby(["Tracker"]).agg({"IoU": "mean"}).reset_index()

# Define success using the same IoU thresholds
df_occ["Success Rate 50"] = (df_occ["IoU"] > 0.5).astype(int)
df_occ["Success Rate 40"] = (df_occ["IoU"] > 0.4).astype(int)
df_occ["Success Rate 30"] = (df_occ["IoU"] > 0.3).astype(int)

# Compute success rate percentages
sucess_rate_occ_df = (
    df.groupby("Tracker")
    .agg({"Success Rate 50": "mean", "Success Rate 40": "mean", "Success Rate 30": "mean"})
    .reset_index()
)

sucess_rate_occ_df["Success Rate 50"] = sucess_rate_occ_df["Success Rate 50"] * 100
sucess_rate_occ_df["Success Rate 40"] = sucess_rate_occ_df["Success Rate 40"] * 100
sucess_rate_occ_df["Success Rate 30"] = sucess_rate_occ_df["Success Rate 30"] * 100

# Create occlusion summary table
summary_occ = pd.DataFrame({
    "Tracker": df_occ["Tracker"],
    "Average IoU under Occlusion": df_occ["IoU"],
    "Success Rate 50 under Occlusion (%)": sucess_rate_occ_df["Success Rate 50"],
    "Success Rate 40 under Occlusion (%)": sucess_rate_occ_df["Success Rate 40"],
    "Success Rate 30 under Occlusion (%)": sucess_rate_occ_df["Success Rate 30"]
})

# Display occlusion results
print(summary_occ)
print("Average IoU under Occlusion:", summary_occ["Average IoU under Occlusion"].mean())
print("Success Rate 50 under Occlusion (%):", summary_occ["Success Rate 50 under Occlusion (%)"].mean())
print("Success Rate 40 under Occlusion (%):", summary_occ["Success Rate 40 under Occlusion (%)"].mean())
print("Success Rate 30 under Occlusion (%):", summary_occ["Success Rate 30 under Occlusion (%)"].mean())

# Save occlusion summary
summary_occ.to_csv("results/summary_occlusion.csv", index=False)


# ===================== NOISE RESULTS ANALYSIS =====================
print("\n###################### Noise Experiment Results Summary ############################")

# Load noise experiment results
df_noise = pd.read_csv('results/noise_results.csv')

# Compute average IoU per tracker under noise
df_noise = df_noise.groupby(["Tracker"]).agg({"IoU": "mean"}).reset_index()

# Define success thresholds
df_noise["Success Rate 50"] = (df_noise["IoU"] > 0.5).astype(int)
df_noise["Success Rate 40"] = (df_noise["IoU"] > 0.4).astype(int)
df_noise["Success Rate 30"] = (df_noise["IoU"] > 0.3).astype(int)

# Compute success rates
sucess_rate_noise_df = (
    df_noise.groupby(["Tracker"])
    .agg({"Success Rate 50": "mean", "Success Rate 40": "mean", "Success Rate 30": "mean"})
    .reset_index()
)

sucess_rate_noise_df["Success Rate 50"] = sucess_rate_noise_df["Success Rate 50"] * 100
sucess_rate_noise_df["Success Rate 40"] = sucess_rate_noise_df["Success Rate 40"] * 100
sucess_rate_noise_df["Success Rate 30"] = sucess_rate_noise_df["Success Rate 30"] * 100

# Create noise summary table
summary_noise = pd.DataFrame({
    "Tracker": df_noise["Tracker"],
    "Average IoU under Noise": df_noise["IoU"],
    "Success Rate 50 under Noise (%)": sucess_rate_noise_df["Success Rate 50"],
    "Success Rate 40 under Noise (%)": sucess_rate_noise_df["Success Rate 40"],
    "Success Rate 30 under Noise (%)": sucess_rate_noise_df["Success Rate 30"]
})

# Display noise results
print(summary_noise)
print("Average IoU under Noise:", summary_noise["Average IoU under Noise"].mean())
print("Success Rate 50 under Noise (%):", summary_noise["Success Rate 50 under Noise (%)"].mean())
print("Success Rate 40 under Noise (%):", summary_noise["Success Rate 40 under Noise (%)"].mean())
print("Success Rate 30 under Noise (%):", summary_noise["Success Rate 30 under Noise (%)"].mean())

# Save noise summary
summary_noise.to_csv("results/summary_noise.csv", index=False)


# ===================== FPS ANALYSIS =====================
print("\n###################### FPS Analysis ############################")

# Load FPS measurements
df_fps = pd.read_csv('results/fps_results.csv')

# Compute average FPS per tracker and experiment
average_fps = df_fps.groupby(["Tracker", "Experiment"]).agg({"FPS": "mean"}).reset_index()

# Create FPS summary table
summary_fps = pd.DataFrame({
    "Tracker": average_fps["Tracker"],
    "Experiment": average_fps["Experiment"],
    "Average FPS": average_fps["FPS"]
})

# Display FPS summary
print(summary_fps)

# Save FPS summary
summary_fps.to_csv("results/summary_fps.csv", index=False)
