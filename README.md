# 🚀 P566 – Customer Segmentation using Clustering

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualizations-lightblue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> *Goal:* Perform *clustering* to summarize customer segments and derive actionable marketing insights.  
> *Dataset:* marketing_campaign.xlsx

---

## *📌 Project Overview*
This project focuses on *segmenting customers* based on their demographic and spending behavior.  
We used *unsupervised machine learning techniques* to identify meaningful customer clusters.

---

## *🧩 Project Workflow*
### *1. Exploratory Data Analysis (EDA)*
- Handled missing values and duplicates
- Detected & removed outliers using *Z-score* & *IQR*
- Derived new features like *Total_Spent*
- Visualized relationships between variables

### *2. Feature Engineering*
- Label encoding for categorical variables
- Scaling numerical features using *StandardScaler*

### *3. Model Building*
- Implemented *K-Means, **Hierarchical Clustering, and **DBSCAN*
- Evaluated using *Elbow Method, **Silhouette Score, and **Davies-Bouldin Index*

### *4. Dimensionality Reduction*
- Applied *PCA* to visualize clusters in *2D and 3D*

### *5. Deployment*
- Exported trained models and scalers (kmeans_model.pkl, scaler.pkl)
- Created a deployment-ready app.py

---

## *📊 Tech Stack*
- *Language:* Python 3.10+
- *Libraries:* Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Plotly
- *Deployment:* Flask / Streamlit

---

## *📂 Project Structure*
