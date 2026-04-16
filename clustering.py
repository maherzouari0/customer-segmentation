import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── 1. Load Data ────────────────────────────────────────────
df = pd.read_csv("customers.csv")

# ── 2. Feature Selection ────────────────────────────────────
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ── 3. Feature Scaling (IMPORTANT for K-Means) ──────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Elbow Method (find optimal k) ────────────────────────
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker="o", linewidth=2)
plt.title("Elbow Method — Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("elbow.png", dpi=150)
plt.show()

# ── 5. Train Final Model (k=5) ──────────────────────────────
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ── 6. Cluster Labels (Business Interpretation) ─────────────
cluster_labels = {
    0: "Low Income / High Spending",
    1: "High Income / High Spending",
    2: "Medium Income / Medium Spending",
    3: "High Income / Low Spending",
    4: "Low Income / Low Spending"
}

df["Segment"] = df["Cluster"].map(cluster_labels)

# ── 7. Plot Clusters ────────────────────────────────────────
colors = ["red", "green", "blue", "orange", "purple"]

plt.figure(figsize=(9, 6))

for cluster_id, label in cluster_labels.items():
    subset = df[df["Cluster"] == cluster_id]
    plt.scatter(
        subset["Annual Income (k$)"],
        subset["Spending Score (1-100)"],
        label=label,
        color=colors[cluster_id],
        s=80,
        alpha=0.8
    )

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=200,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segments — K-Means Clustering (k=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("clusters.png", dpi=150)
plt.show()

# ── 8. Segment Summary ──────────────────────────────────────
print("\n📊 Cluster Summary:")
print(df.groupby("Segment")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1))

# ── 9. Cluster Centers (Original Scale) ─────────────────────
print("\n📍 Cluster Centers (Original Scale):")
print(pd.DataFrame(centers, columns=["Annual Income", "Spending Score"]).round(1))
