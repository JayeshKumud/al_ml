# %%
# ------------------------------------------------------------
# MODEL & ALGORITHM DOCUMENTATION — K-MEANS CLUSTERING
# ------------------------------------------------------------
# MODEL NAME:
#   KMeans (Unsupervised Machine Learning Algorithm)
#
# MODEL DEFINITION:
#   K-Means is a clustering algorithm that groups data points
#   into K clusters based on similarity. It is unsupervised,
#   meaning it does not use labeled output. Instead, it discovers
#   natural patterns in the data by minimizing the distance
#   between points and their assigned cluster center (centroid).
#
# MODEL FUNCTIONING (HOW IT WORKS INTERNALLY):
#
#   1. Choose the number of clusters (K).
#
#   2. Randomly initialize K centroids.
#      These centroids represent the initial cluster centers.
#
#   3. Assign each data point to the nearest centroid.
#      Distance metric: Euclidean distance.
#
#   4. Recalculate centroids:
#      Each centroid becomes the mean of all points assigned
#      to that cluster.
#
#   5. Repeat steps 3–4 until:
#        • Centroids stop moving (convergence), OR
#        • Maximum iterations reached.
#
#   The final result:
#        • Cluster labels for each data point
#        • Final centroid positions
#
# ------------------------------------------------------------
# FEATURES USED IN THIS CODE:
#
#   Age
#   Income($)
#
# WHY THESE FEATURES:
#   Both are continuous numeric values, ideal for distance-based
#   clustering. They help identify groups of people with similar
#   age-income patterns.
#
# ------------------------------------------------------------
# WHY SCALING IS IMPORTANT:
#
#   K-Means uses Euclidean distance.
#   If one feature has a larger numeric range (Income),
#   it dominates the distance calculation.
#
#   MinMaxScaler normalizes all features to the same range (0–1),
#   ensuring both features contribute equally.


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# %%
# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv("./data/13_k_mean_model_income.csv")

# Display first few rows
df.head()

# %%
# ------------------------------------------------------------
# SCATTER PLOT (RAW DATA)
# ------------------------------------------------------------
plt.scatter(df.Age, df["Income($)"])
plt.xlabel("Age")
plt.ylabel("Income ($)")
plt.title("Raw Data Distribution")
plt.show()

# %%
# ------------------------------------------------------------
# K-MEANS MODEL (WITHOUT SCALING)
# ------------------------------------------------------------
# n_clusters = 3 → We want 3 groups
km = KMeans(n_clusters=3)

# fit_predict() returns cluster labels for each row
y_predicted = km.fit_predict(df[["Age", "Income($)"]])

print("Cluster Labels (Unscaled):", y_predicted)

# Add cluster labels to DataFrame
df["cluster"] = y_predicted

# Display updated DataFrame
df.head()

# %%
# ------------------------------------------------------------
# CLUSTER CENTERS (WITHOUT SCALING)
# ------------------------------------------------------------
print("Cluster Centers (Unscaled):")
print(km.cluster_centers_)

# %%
# ------------------------------------------------------------
# VISUALIZE CLUSTERS (WITHOUT SCALING)
# ------------------------------------------------------------
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1["Income($)"], color="green", label="Cluster 0")
plt.scatter(df2.Age, df2["Income($)"], color="red", label="Cluster 1")
plt.scatter(df3.Age, df3["Income($)"], color="black", label="Cluster 2")

# Plot centroids
plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    color="purple",
    marker="*",
    s=200,
    label="Centroids",
)

plt.xlabel("Age")
plt.ylabel("Income ($)")
plt.legend()
plt.title("Clusters Without Scaling")
plt.show()

# %%
# ------------------------------------------------------------
# PREPROCESSING USING MIN-MAX SCALER
# ------------------------------------------------------------
scaler = MinMaxScaler()

# Scale Income
df["Income($)"] = scaler.fit_transform(df[["Income($)"]])

# Scale Age
df["Age"] = scaler.fit_transform(df[["Age"]])

# Display scaled DataFrame
df.head()

# %%
# ------------------------------------------------------------
# K-MEANS MODEL (WITH SCALING)
# ------------------------------------------------------------
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[["Age", "Income($)"]])

df["cluster"] = y_predicted

print("Cluster Labels (Scaled):", y_predicted)
df.head()

# %%
# ------------------------------------------------------------
# CLUSTER CENTERS (SCALED)
# ------------------------------------------------------------
print("Cluster Centers (Scaled):")
print(km.cluster_centers_)

# %%
# ------------------------------------------------------------
# VISUALIZE CLUSTERS (WITH SCALING)
# ------------------------------------------------------------
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1["Income($)"], color="green", label="Cluster 0")
plt.scatter(df2.Age, df2["Income($)"], color="red", label="Cluster 1")
plt.scatter(df3.Age, df3["Income($)"], color="black", label="Cluster 2")

plt.scatter(
    km.cluster_centers_[:, 0],
    km.cluster_centers_[:, 1],
    color="purple",
    marker="*",
    s=200,
    label="Centroids",
)

plt.xlabel("Age (Scaled)")
plt.ylabel("Income ($) (Scaled)")
plt.legend()
plt.title("Clusters With Scaling")
plt.show()

# %%
# ------------------------------------------------------------
# ELBOW PLOT — FIND BEST K
# ------------------------------------------------------------
sse = []  # Sum of squared errors
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[["Age", "Income($)"]])
    sse.append(km.inertia_)  # inertia_ = SSE

plt.plot(k_range, sse, marker="o")
plt.xlabel("K (Number of Clusters)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method For Optimal K")
plt.show()
