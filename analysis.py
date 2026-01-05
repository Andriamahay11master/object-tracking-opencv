#------------------------compute average IoU and success rate-----------------------------------------
import pandas as pd

#------------------------ LOAD BASELINE RESULTS -----------------------------------------
print("######################Baseline Experiment Results Summary############################")
df = pd.read_csv('results/baseline_results.csv')
avg_iou = df.groupby(["Tracker"]).agg({"IoU": "mean"}).reset_index()
df["Success Rate 50"] = df["IoU"].apply(lambda x: 1 if x > 0.5 else 0)
df["Success Rate 30"] = df["IoU"].apply(lambda x: 1 if x > 0.3 else 0)
sucess_rate_df = df.groupby(["Tracker"]).agg({"Success Rate 50": "mean", "Success Rate 30": "mean"}).reset_index() * 100
summary = pd.DataFrame({
    "Tracker": avg_iou["Tracker"],
    "Average IoU": avg_iou["IoU"],
    "Success Rate 50 (%)": sucess_rate_df["Success Rate 50"],
    "Success Rate 30 (%)": sucess_rate_df["Success Rate 30"]
})
print(summary)
print("Average IoU", summary["Average IoU"].mean())
print("Success Rate 50 (%)", summary["Success Rate 50 (%)"].mean())
print("Success Rate 30 (%)", summary["Success Rate 30 (%)"].mean())
summary.to_csv("results/summary_baseline.csv", index=False)

#------------------------ LOAD OCCLUSION RESULTS -----------------------------------------
print("\n######################Occlusion Experiment Results Summary############################")
df_occ = pd.read_csv('results/occlusion_results.csv')
df_occ = df_occ.groupby(["Tracker"]).agg({"IoU": "mean"}).reset_index()
df_occ["Success Rate 50"] = df_occ["IoU"].apply(lambda x: 1 if x > 0.5 else 0)
df_occ["Success Rate 30"] = df_occ["IoU"].apply(lambda x: 1 if x > 0.3 else 0)
sucess_rate_occ_df = df_occ.groupby(["Tracker"]).agg({"Success Rate 50": "mean", "Success Rate 30": "mean"}).reset_index() * 100
summary_occ = pd.DataFrame({
    "Tracker": df_occ["Tracker"],
    "Average IoU under Occlusion": df_occ["IoU"],
    "Success Rate 50 under Occlusion (%)": sucess_rate_occ_df["Success Rate 50"],
    "Success Rate 30 under Occlusion (%)": sucess_rate_occ_df["Success Rate 30"]
})
print(summary_occ)
print("Average IoU under Occlusion", summary_occ["Average IoU under Occlusion"].mean())
print("Success Rate 50 under Occlusion (%)", summary_occ["Success Rate 50 under Occlusion (%)"].mean())
print("Success Rate 30 under Occlusion (%)", summary_occ["Success Rate 30 under Occlusion (%)"].mean())
summary_occ.to_csv("results/summary_occlusion.csv", index=False)

#------------------------ LOAD NOISE RESULTS -----------------------------------------
print("\n######################Noise Experiment Results Summary############################")
df_noise = pd.read_csv('results/noise_results.csv')
df_noise = df_noise.groupby(["Tracker"]).agg({"IoU": "mean"}).reset_index()
df_noise["Success Rate 50"] = df_noise["IoU"].apply(lambda x: 1 if x > 0.5 else 0)
df_noise["Success Rate 30"] = df_noise["IoU"].apply(lambda x: 1 if x > 0.3 else 0)
sucess_rate_noise_df = df_noise.groupby(["Tracker"]).agg({"Success Rate 50": "mean", "Success Rate 30": "mean"}).reset_index() * 100
summary_noise = pd.DataFrame({
    "Tracker": df_noise["Tracker"],
    "Average IoU under Noise": df_noise["IoU"], 
    "Success Rate 50 under Noise (%)": sucess_rate_noise_df["Success Rate 50"],
    "Success Rate 30 under Noise (%)": sucess_rate_noise_df["Success Rate 30"]
})
print(summary_noise)
print("Average IoU under Noise", summary_noise["Average IoU under Noise"].mean())
print("Success Rate 50 under Noise (%)", summary_noise["Success Rate 50 under Noise (%)"].mean())
print("Success Rate 30 under Noise (%)", summary_noise["Success Rate 30 under Noise (%)"].mean())
summary_noise.to_csv("results/summary_noise.csv", index=False)


#------------------------ FPS ANALYSIS -----------------------------------------
print("\n######################FPS Analysis############################")
df_fps = pd.read_csv('results/fps_results.csv')
average_fps = df_fps.groupby(["Tracker", "Experiment"]).agg({"FPS": "mean"}).reset_index()
summary_fps = pd.DataFrame({
    "Tracker": average_fps["Tracker"],
    "Experiment": average_fps["Experiment"],
    "Average FPS": average_fps["FPS"]
})
print(summary_fps)
summary_fps.to_csv("results/summary_fps.csv", index=False)