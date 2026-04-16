# 📊 Customer Segmentation using K-Means Clustering

Unsupervised machine learning project that segments customers 
based on income and spending behavior to support targeted 
marketing strategies.

---

## 🎯 Objective

Identify distinct customer profiles from a retail dataset 
using K-Means clustering — turning raw behavioral data into 
actionable business segments.

---

## 📁 Dataset

- **Source:** Mall Customers Dataset (200 records)
- **Features used:** Annual Income (k$), Spending Score (1–100)

---

## 🧠 Methodology

1. **Data Loading & Exploration** — inspect shape, types, nulls
2. **Feature Selection** — income and spending score as clustering features
3. **Elbow Method** — determine optimal number of clusters (k=5)
4. **K-Means Clustering** — train final model and assign segment labels
5. **Visualization** — scatter plot with color-coded segments
6. **Segment Analysis** — mean income and spending per cluster

---

## 📈 Results

| Segment | Avg Income | Avg Spending Score |
|---|---|---|
| High Income / High Spending | ~87k | ~82 |
| High Income / Low Spending | ~88k | ~18 |
| Medium Income / Medium Spending | ~55k | ~50 |
| Low Income / High Spending | ~26k | ~78 |
| Low Income / Low Spending | ~27k | ~20 |

---

## 📊 Visual Results

### Elbow Method
![Elbow](elbow.png)

### Customer Clusters
![Clusters](clusters.png)

---

## 💡 Business Value

- **High Income / High Spending** → Premium loyalty programs
- **Low Income / High Spending** → Risk of churn, needs retention offers
- **High Income / Low Spending** → Untapped potential, needs engagement campaigns
- Segments provide a clear foundation for data-driven marketing decisions

---

## 🛠 Tools & Libraries

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-learn | K-Means clustering |
| Matplotlib | Visualization |

---

## 🚀 How to Run

```bash
pip install pandas scikit-learn matplotlib
python clustering.py
```

## 🛠 Tools
Python, Pandas, Scikit-learn, Matplotlib
