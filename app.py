import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_ton_iot
from gnn_model import GCNAnomalyDetector, build_graph
import torch

st.title("🔐 GNN-Based Cybersecurity Anomaly Detection")

uploaded = st.file_uploader("Upload TON_IoT or CICIDS CSV", type="csv")

if uploaded:
    df = preprocess_ton_iot(uploaded)
    st.success("File successfully processed.")
    
    data = build_graph(df)

    model = GCNAnomalyDetector(input_dim=df.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.NLLLoss()

    # Dummy labels (normal = 0, anomaly = 1), replace with real ones if available
    y = torch.randint(0, 2, (df.shape[0],))
    data.y = y

    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)

    # Show results
    st.subheader("📊 Prediction Summary")
    normal = (pred == 0).sum().item()
    anomaly = (pred == 1).sum().item()
    st.write(f"✅ Normal: {normal} | ⚠️ Anomaly: {anomaly}")

    fig, ax = plt.subplots()
    sns.countplot(x=pred.numpy(), ax=ax)
    ax.set_title("Prediction Distribution (0=Normal, 1=Anomaly)")
    st.pyplot(fig)