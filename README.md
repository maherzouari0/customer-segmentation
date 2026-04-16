# 📊 Customer Segmentation using K-Means Clustering

Unsupervised machine learning project that segments customers based on income and spending behavior to support data-driven marketing strategies.

---

## 🎯 Objective

Identify distinct customer segments from behavioral data to enable targeted marketing and support business decision-making.

---

## 📁 Dataset

- **Source:** Mall Customers Dataset (200 records)  
- **Features used:** Annual Income (k$), Spending Score (1–100)

---

## 🧠 Methodology

1. **Data Loading & Exploration** — inspected dataset structure and quality  
2. **Feature Selection** — selected income and spending score as clustering variables  
3. **Feature Scaling** — standardized features using StandardScaler for optimal distance-based clustering  
4. **Elbow Method** — determined optimal number of clusters using inertia minimization (k=5)  
5. **K-Means Clustering** — trained model and assigned cluster labels  
6. **Visualization** — created scatter plots to visualize customer segments  
7. **Segment Analysis** — analyzed average income and spending behavior per cluster  

---

## ⚙️ Technical Highlights

- Implemented feature scaling to improve clustering performance  
- Applied Elbow Method to determine optimal cluster count  
- Used K-Means clustering for unsupervised learning  
- Interpreted clusters with business-oriented labels for real-world usability  

---

## 📈 Results

| Segment | Avg Income | Avg Spending Score |
|--------|-----------|--------------------|
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

- Enables targeted marketing strategies based on customer behavior  
- Identifies high-value customer segments for retention and loyalty programs  
- Highlights underperforming segments for engagement and growth opportunities  
- Supports data-driven decision-making for revenue optimization  

---

## 🛠 Tools & Libraries

| Tool | Purpose |
|------|--------|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| Scikit-learn | K-Means clustering |
| Matplotlib | Data visualization |

---

## 🚀 How to Run

```bash
pip install pandas scikit-learn matplotlib
python clustering.py
