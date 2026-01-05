import pandas as pd
import matplotlib.pyplot as plt

#Load summary results
df_baseline = pd.read_csv('results/summary_baseline.csv')
df_occlusion = pd.read_csv('results/summary_occlusion.csv')
df_noise = pd.read_csv('results/summary_noise.csv')

#Plot bar chart for baseline average IoU
plt.figure(figsize=(10,6))
bars = plt.bar(df_baseline["Tracker"], df_baseline["Average IoU"], color=['green', 'purple', 'orange'])
plt.title("Average IoU – Baseline")
plt.ylabel("Average IoU")
plt.xlabel("Tracker")
plt.ylim(0, 1)
plt.show()

#Plot bar chart for occlusion average IoU
plt.figure(figsize=(10,6))
bars = plt.bar(df_occlusion["Tracker"], df_occlusion["Average IoU under Occlusion"], color=['green', 'purple', 'orange'])
plt.title("Average IoU – Occlusion")
plt.ylabel("Average IoU")
plt.xlabel("Tracker")
plt.ylim(0, 1)
plt.show()

#Plot bar chart for noise average IoU
plt.figure(figsize=(10,6))
bars = plt.bar(df_noise["Tracker"], df_noise["Average IoU under Noise"], color=['green', 'purple', 'orange'])
plt.title("Average IoU – Noise")
plt.ylabel("Average IoU")
plt.xlabel("Tracker")
plt.ylim(0, 1)
plt.show()


#Plot average IoU comparison
plt.figure(figsize=(10,6))
plt.plot(df_baseline["Tracker"], df_baseline["Average IoU"], label="Baseline", marker='o')
plt.plot(df_occlusion["Tracker"], df_occlusion["Average IoU under Occlusion"], label="Occlusion", marker='s')
plt.plot(df_noise["Tracker"], df_noise["Average IoU under Noise"], label="Noise", marker='^')
plt.xlabel("Tracker")
plt.ylabel("Average IoU")
plt.title("Average IoU Comparison")
plt.legend()
plt.show()

#Plot sucess rate comparison
plt.figure(figsize=(10,6))
plt.plot(df_baseline["Tracker"], df_baseline["Success Rate 50 (%)"], label="Baseline")
plt.plot(df_occlusion["Tracker"], df_occlusion["Success Rate 50 under Occlusion (%)"], label="Occlusion")
plt.plot(df_noise["Tracker"], df_noise["Success Rate 50 under Noise (%)"], label="Noise")
plt.xlabel("Tracker")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate Comparison")
plt.legend()
plt.show()