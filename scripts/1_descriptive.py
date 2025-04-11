import pandas as pd
import yaml, os
import numpy as np
import seaborn as sns
import warnings
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Or try 'Qt5Agg', depending on your setup
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

######## Notes: Correct path. Output intermediate files. Github.
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Access the raw data directory
data_dir = config["raw_data"]["nhts"]
processed_dir = config["processed_data"]["nhts"]

adjweights = pd.read_csv(os.path.join(data_dir,"2017","county_wt_adj.csv"))

########### Read trip chain: 2017
chain = pd.read_csv(os.path.join(data_dir,"2017","chntrp17.csv"))

# number of stops distribution
stop_summary = chain.groupby(["HOUSEID", "PERSONID", "TOUR"]).size().reset_index(name="count")
stop_summary['num_stops'] = stop_summary["count"]-1

sns.histplot(data = stop_summary, x = 'num_stops', bins=60)
plt.xlim(0, 10)  # Zoom into the 0–5 range
plt.show()

# purpose distribution
sequences = chain.groupby(["HOUSEID", "PERSONID"]).agg({
    "TOURTYPE": lambda x: '-'.join(map(str, x)),
    "WTTRDFIN": "first"  # assuming weight is the same for all trips of a person
}).reset_index()
# Step 2: Rename for clarity
sequences.rename(columns={"TOURTYPE": "sequence", "WTTRDFIN": "weight"}, inplace=True)
# Step 3: Group by unique sequence and sum the weights
sequence_counts = sequences.groupby("sequence")["weight"].sum().reset_index(name="weighted_count")
# Optional: sort by most common
sequence_counts = sequence_counts.sort_values(by="weighted_count", ascending=False)
sequence_counts["weighted_share"] = sequence_counts["weighted_count"]/sequence_counts["weighted_count"].sum()*100

top5 = sequence_counts.head(5)
plt.figure(figsize=(8, 5))
plt.barh(top5["sequence"], top5["weighted_share"])
plt.xlabel("Weighted Share")
plt.ylabel("Trip Purpose Sequence")
plt.title("Top 5 Most Common Trip Purpose Sequences (Weighted)")
plt.gca().invert_yaxis()  # optional: put highest on top
plt.tight_layout()
plt.show()