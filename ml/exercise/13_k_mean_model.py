# %%
# MODEL DEFINITION:
#   K-Means is a clustering algorithm that groups data points
#   into K clusters based on similarity. It is unsupervised,
#   meaning it does not use labeled output. Instead, it discovers
#   natural patterns in the data by minimizing the distance
#   between points and their assigned cluster center (centroid).
# ------------------------------------------------------------
# K-MEANS CLUSTERING — IRIS DATASET (DOCUMENTED VERSION)
# ------------------------------------------------------------
# Model Used: KMeans (Unsupervised ML)
# Goal: Form clusters of iris flowers using petal length & width
# Features Used: petal length (cm), petal width (cm)
# ------------------------------------------------------------
# ALGORITHM EXPLANATION:
# 1. Choose number of clusters (K)
# 2. Randomly initialize K centroids
# 3. Assign each data point to nearest centroid (Euclidean distance)
# 4. Recompute centroids as mean of assigned points
# 5. Repeat until centroids stop moving (convergence)
#
# WHY SCALING MAY HELP:
# K-Means is distance-based. If features have different scales,
# larger-scale features dominate. MinMaxScaler normalizes values
# to 0–1 range, improving clustering.
#
# OUTPUTS:
# • Cluster labels
# • Centroid coordinates
# • Cluster visualization
# • Elbow plot to find optimal K
# ------------------------------------------------------------

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

# %%
# ------------------------------------------------------------
# LOAD IRIS DATASET
# ------------------------------------------------------------
iris = load_iris()

# Create DataFrame from iris features
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display first few rows
df.head()

# %%
# ------------------------------------------------------------
# ADD TARGET COLUMN (flower species)
# ------------------------------------------------------------
df["flower"] = iris.target
df.head()

# %%
# ------------------------------------------------------------
# DROP UNUSED FEATURES (ONLY PETAL LENGTH & WIDTH)
# ------------------------------------------------------------
df.drop(
    ["sepal length (cm)", "sepal width (cm)", "flower"], axis="columns", inplace=True
)

# Display cleaned DataFrame
df.head(3)

# %%
# ------------------------------------------------------------
# K-MEANS MODEL (WITHOUT SCALING)
# ------------------------------------------------------------
km = KMeans(n_clusters=3)

# fit_predict returns cluster labels
yp = km.fit_predict(df)

print("Cluster Labels:", yp)

# Add cluster labels to DataFrame
df["cluster"] = yp

# Display updated DataFrame
df.head(2)

# %%
# ------------------------------------------------------------
# UNIQUE CLUSTER VALUES
# ------------------------------------------------------------
print("Unique Clusters:", df.cluster.unique())

# %%
# ------------------------------------------------------------
# VISUALIZE CLUSTERS (WITHOUT SCALING)
# ------------------------------------------------------------
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(
    df1["petal length (cm)"], df1["petal width (cm)"], color="blue", label="Cluster 0"
)
plt.scatter(
    df2["petal length (cm)"], df2["petal width (cm)"], color="green", label="Cluster 1"
)
plt.scatter(
    df3["petal length (cm)"], df3["petal width (cm)"], color="yellow", label="Cluster 2"
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Clusters (Without Scaling)")
plt.legend()
plt.show()

# %%
# ------------------------------------------------------------
# OPTIONAL: PREPROCESSING USING MIN-MAX SCALER
# ------------------------------------------------------------
scaler = MinMaxScaler()

df_scaled = df[["petal length (cm)", "petal width (cm)"]].copy()
df_scaled["petal length (cm)"] = scaler.fit_transform(df[["petal length (cm)"]])
df_scaled["petal width (cm)"] = scaler.fit_transform(df[["petal width (cm)"]])

# Display scaled DataFrame
df_scaled.head()

# %%
# ------------------------------------------------------------
# K-MEANS MODEL (WITH SCALING)
# ------------------------------------------------------------
km_scaled = KMeans(n_clusters=3)
yp_scaled = km_scaled.fit_predict(df_scaled)

print("Cluster Labels (Scaled):", yp_scaled)

df_scaled["cluster"] = yp_scaled
df_scaled.head()

# %%
# ------------------------------------------------------------
# VISUALIZE CLUSTERS (WITH SCALING)
# ------------------------------------------------------------
df1 = df_scaled[df_scaled.cluster == 0]
df2 = df_scaled[df_scaled.cluster == 1]
df3 = df_scaled[df_scaled.cluster == 2]

plt.scatter(
    df1["petal length (cm)"], df1["petal width (cm)"], color="blue", label="Cluster 0"
)
plt.scatter(
    df2["petal length (cm)"], df2["petal width (cm)"], color="green", label="Cluster 1"
)
plt.scatter(
    df3["petal length (cm)"], df3["petal width (cm)"], color="yellow", label="Cluster 2"
)

plt.xlabel("Petal Length (Scaled)")
plt.ylabel("Petal Width (Scaled)")
plt.title("Iris Clusters (With Scaling)")
plt.legend()
plt.show()

# %%
# ------------------------------------------------------------
# ELBOW PLOT — FIND BEST K
# ------------------------------------------------------------
sse = []
k_rng = range(1, 10)

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df_scaled[["petal length (cm)", "petal width (cm)"]])
    sse.append(km.inertia_)  # inertia_ = SSE

plt.plot(k_rng, sse, marker="o")
plt.xlabel("K (Number of Clusters)")
plt.ylabel("SSE (Inertia)")
plt.title("Elbow Method — Optimal K")
plt.show()
