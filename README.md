# 🔗 GNN-Based Cybersecurity Anomaly Detection

A Streamlit-powered web application that uses **Graph Neural Networks (GNN)** to detect anomalies in cybersecurity network traffic. Built with **PyTorch Geometric**, the app allows live training and visualization of network-based threat patterns.

---

## 🚀 Features

- 📂 Upload TON_IoT or CICIDS datasets
- 🔗 Build graph using correlation-based structure
- 🧠 Train GCN model on uploaded data
- 📊 Live anomaly prediction (Normal vs Anomaly)
- 📈 Interactive plots with Seaborn + Matplotlib

---

## 🧠 GNN Model Architecture

```text
Input features → GCNConv(16) → ReLU → GCNConv(2) → LogSoftmax
```
The model is defined in gnn_model.py using PyTorch Geometric.
Graph is built from the correlation matrix of input features using dense_to_sparse.


▶️ Run Locally

- git clone https://github.com/Nisha-Asabe/cyber-anomaly-detection.git
- cd cyber-anomaly-detection
- pip install -r requirements.txt
- streamlit run app.py

