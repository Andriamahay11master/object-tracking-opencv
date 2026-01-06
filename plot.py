import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# This script generates visualizations of tracking performance
# based on the summarized evaluation results. It plots
# average IoU and success rate comparisons across trackers
# and experimental conditions.
# ------------------------------------------------------------

# Load summarized evaluation results
df_baseline = pd.read_csv('results/summary_baseline.csv')
df_occlusion = pd.read_csv('results/summary_occlusion.csv')
df_noise = pd.read_csv('results/summary_noise.csv')

# ===================== BASELINE IoU BAR PLOT =====================
# Plot average IoU for each tracker under baseline conditions
plt.figure(figsize=(10, 6))
plt.bar(
    df_baseline["Tracker"],
    df_baseline["Average IoU"],
    color=['green', 'purple', 'orange','black']
)
plt.title("Average IoU – Baseline")
plt.ylabel("Average IoU")
plt.xlabel("Tracker")
plt.ylim(0, 1)
plt.show()

# ===================== OCCLUSION IoU BAR PLOT =====================
# Plot average IoU for each tracker under occlusion conditions
plt.figure(figsize=(10, 6))
plt.bar(
    df_occlusion["Tracker"],
    df_occlusion["Average IoU under Occlusion"],
    color=['green', 'purple', 'orange','black']
)
plt.title("Average IoU – Occlusion")
plt.ylabel("Average IoU")
plt.xlabel("Tracker")
plt.ylim(0, 1)
plt.show()

# ===================== NOISE IoU BAR PLOT =====================
# Plot average IoU for each tracker under noise conditions
plt.figure(figsize=(10, 6))
plt.bar(
    df_noise["Tracker"],
    df_noise["Average IoU under Noise"],
    color=['green', 'purple', 'orange','black']
)
plt.title("Average IoU – Noise")
plt.ylabel("Average IoU")
plt.xlabel("Tracker")
plt.ylim(0, 1)
plt.show()


# ===================== AVERAGE IoU COMPARISON =====================
# Line plot comparing average IoU across all experimental conditions
plt.figure(figsize=(10, 6))
plt.plot(
    df_baseline["Tracker"],
    df_baseline["Average IoU"],
    label="Baseline",
    marker='o'
)
plt.plot(
    df_occlusion["Tracker"],
    df_occlusion["Average IoU under Occlusion"],
    label="Occlusion",
    marker='s'
)
plt.plot(
    df_noise["Tracker"],
    df_noise["Average IoU under Noise"],
    label="Noise",
    marker='^'
)
plt.xlabel("Tracker")
plt.ylabel("Average IoU")
plt.title("Average IoU comparison across conditions")
plt.legend()
plt.show()


# ===================== SUCCESS RATE COMPARISON =====================
# Line plot comparing success rate (IoU > 0.5) across conditions
x = np.arange(len(df_baseline["Tracker"]))
width = 0.1  # small horizontal offset

plt.figure(figsize=(10, 6))

plt.plot(
    x - width,
    df_baseline["Success Rate 50 (%)"],
    label="Baseline",
    marker='o'
)

plt.plot(
    x,
    df_occlusion["Success Rate 50 under Occlusion (%)"],
    label="Occlusion",
    marker='s'
)

plt.plot(
    x + width,
    df_noise["Success Rate 50 under Noise (%)"],
    label="Noise",
    marker='^'
)

plt.xticks(x, df_baseline["Tracker"])
plt.xlabel("Tracker")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate 50 comparison across conditions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
