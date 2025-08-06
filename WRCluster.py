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

nfl = pd.read_csv('play_by_play_2024.csv')

receiving_plays = nfl[
    (nfl['play_type'] == 'pass') &
    (nfl['receiver_player_name'].notna() &
    (nfl['season_type'] == 'REG'))
]

# Aggregate receiver stats
receiver_stats = receiving_plays.groupby(['receiver_player_name']).agg({
    'complete_pass': 'sum',
    'pass_attempt': 'sum',
    'passing_yards': 'sum',
    'yards_after_catch': 'sum',
    'air_yards': 'sum',
    'touchdown': 'sum',
    'epa': 'sum'
}).reset_index()

# Rename and calculate key metrics
receiver_stats.rename(columns={
    'receiver_player_name': 'Receiver',
    'complete_pass': 'Receptions',
    'pass_attempt': 'Targets',
    'passing_yards': 'Receiving_Yards',
    'yards_after_catch': 'YAC',
    'air_yards': 'Air_Yards',
    'touchdown': 'TDs',
    'epa': 'Total_EPA'
}, inplace=True)

receiver_stats['Catch_Rate'] = receiver_stats['Receptions'] / receiver_stats['Targets']
receiver_stats['Yards_per_Target'] = receiver_stats['Receiving_Yards'] / receiver_stats['Targets']
receiver_stats['YAC_per_Catch'] = receiver_stats['YAC'] / receiver_stats['Receptions']
receiver_stats['EPA_per_Target'] = receiver_stats['Total_EPA'] / receiver_stats['Targets']

# Filter for players with at least 50 targets
receiver_stats = receiver_stats[receiver_stats['Targets'] >= 50]

st.set_page_config(page_title="NFL WR Clustering Dashboard", layout="wide")
st.title("📊 2024 NFL Wide Receiver Archetype Clusters")

# Load WR stats from existing variable or file
if "receiver_stats" in st.session_state:
    df = st.session_state["receiver_stats"]
else:
    uploaded_file = st.file_uploader("play_by_play_2024.csv", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["receiver_stats"] = df
    else:
        st.info("receiver_stats")
        st.stop()

# Features for clustering
features = [
    "Targets", 
    "Receptions", 
    "Receiving_Yards", 
    "TDs", 
    "EPA_per_Target", 
    "Air_Yards", 
    "YAC_per_Catch"
]

# Drop NA rows
df_clean = receiver_stats.dropna(subset=features + ["Receiver"])
X = df_clean[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar options
n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=6, value=4)

# KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_clean["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df_clean["PCA1"] = coords[:, 0]
df_clean["PCA2"] = coords[:, 1]

# Color palette
palette = sns.color_palette("Set2", n_colors=n_clusters)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.scatterplot(
    data=df_clean,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette=palette,
    s=100,
    ax=ax,
    legend="full"
)

for _, row in df_clean.iterrows():
    ax.text(row["PCA1"], row["PCA2"], row["Receiver"], fontsize=8, alpha=0.75)

ax.set_title("WR Clusters based on Role & Efficiency")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.grid(True)

st.pyplot(fig)

# Show data
with st.expander("🔍 View clustered data"):
    st.dataframe(df_clean.sort_values("Cluster"))


# In[ ]:




