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
The model is defined in gnn_model.py using PyTorch Geometric.

Graph is built from the correlation matrix of input features using dense_to_sparse.

```
▶️ Run Locally

git clone https://github.com/Nisha-Asabe/cyber-anomaly-detection.git
cd cyber-anomaly-detection
pip install -r requirements.txt
streamlit run app.py
🌐 Live Demo
🔗 Click to View Live App
(Update this link after Streamlit deployment)

🧾 Requirements
Python ≥ 3.8

torch, torch-geometric, pandas, streamlit, etc.

txt
Copy
Edit
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.3.2
matplotlib==3.7.4
seaborn==0.12.2
torch==2.2.0
torch-geometric==2.5.1
networkx
