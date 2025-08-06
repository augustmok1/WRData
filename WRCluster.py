#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os


st.set_page_config(page_title="NFL WR Clustering Dashboard", layout="wide")
st.title("📊 NFL Wide Receiver Archetype Clusters Across Seasons")

# Allow user to select season
data_folder = "data"
available_seasons = [f.split("_")[3].split(".")[0] for f in os.listdir(data_folder) if f.startswith("play_by_play")]
season = st.sidebar.selectbox("Select Season", sorted(available_seasons, reverse=True))

# Load selected season data
@st.cache_data
def load_data(season):
    file_path = os.path.join(data_folder, f"play_by_play_{season}.csv")
    return pd.read_csv(file_path)

df = load_data(season)

# Filter to passing plays and receivers only
receiving_plays = df[
    (df["play_type"] == "pass") &
    (df["receiver_player_name"].notna()) &
    (df["season_type"] == "REG")
]

# Aggregate receiver stats
receiver_stats = receiving_plays.groupby("receiver_player_name").agg({
    "complete_pass": "sum",
    "incomplete_pass": "sum",
    "interception": "sum",
    "yards_after_catch": "sum",
    "air_yards": "sum",
    "epa": "mean",
    "yards_gained": "sum",
    "pass": "count"
}).reset_index()

receiver_stats.rename(columns={
    "receiver_player_name": "Receiver",
    "epa": "EPA_per_Target",
    "air_yards": "Air_Yards",
    "yards_after_catch": "YAC",
    "yards_gained": "Receiving_Yards",
    "pass": "Targets"
}, inplace=True)

receiver_stats["YAC_per_Catch"] = receiver_stats["Receiving_Yards"] / receiver_stats["Targets"]
receiver_stats["TDs"] = receiving_plays.groupby("receiver_player_name")["touchdown"].sum().values
receiver_stats["Receptions"] = receiver_stats["complete_pass"]

receiver_stats = receiver_stats[receiver_stats['Targets'] >= 50]

# Drop missing values
features = [
    "Targets",
    "Receptions",
    "Receiving_Yards",
    "TDs",
    "EPA_per_Target",
    "Air_Yards",
    "YAC_per_Catch"
]

df_clean = receiver_stats.dropna(subset=features).copy()
X = df_clean[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=6, value=4)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_clean["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df_clean["PCA1"] = coords[:, 0]
df_clean["PCA2"] = coords[:, 1]

# Plot
st.subheader(f"📌 {season} Receiver Clustering Visualization")
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    data=df_clean,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="Set2",
    s=100,
    ax=ax
)

for _, row in df_clean.iterrows():
    ax.text(row["PCA1"], row["PCA2"], row["Receiver"], fontsize=8, alpha=0.7)

ax.set_title("WR Clusters by Role & Efficiency")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.grid(True)

st.pyplot(fig)

# View data table
with st.expander("📄 View Clustered Receiver Data"):
    st.dataframe(df_clean.sort_values("Cluster"))


# In[ ]:




