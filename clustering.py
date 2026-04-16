import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ── 1. Load Data ────────────────────────────────────────────
df = pd.read_csv("customers.csv")

# ── 2. Feature Selection ────────────────────────────────────
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# ── 3. Elbow Method (find optimal k) ────────────────────────
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia, marker="o", color="steelblue", linewidth=2)
plt.title("Elbow Method — Optimal Number of Clusters", fontsize=13)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("elbow.png", dpi=150)
plt.show()

# ── 4. Train Final Model (k=5) ──────────────────────────────
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# ── 5. Cluster Labels ───────────────────────────────────────
cluster_labels = {
    0: "Low Income / High Spending",
    1: "High Income / High Spending",
    2: "Medium Income / Medium Spending",
    3: "High Income / Low Spending",
    4: "Low Income / Low Spending"
}
df["Segment"] = df["Cluster"].map(cluster_labels)

# ── 6. Plot Clusters ────────────────────────────────────────
colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
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

plt.title("Customer Segments — K-Means Clustering (k=5)", fontsize=13)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("clusters.png", dpi=150)
plt.show()

# ── 7. Segment Summary ──────────────────────────────────────
print("\n📊 Cluster Summary:")
print(df.groupby("Segment")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1))
