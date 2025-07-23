# AI for Cybersecurity Anomaly Detection with Streamlit Dashboard and Autoencoder

# ----------- STEP 1: IMPORTS -----------
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ----------- STEP 2: STREAMLIT UI -----------
st.set_page_config(page_title="Cybersecurity AI Anomaly Detection", layout="wide")
st.title("🔐 Cybersecurity Anomaly Detection System")
uploaded_file = st.file_uploader("📁 Upload Preprocessed Network Traffic CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # ----------- STEP 3: PREPROCESSING -----------
    st.subheader("🧾 Data Overview")
    st.write("First 5 rows of the uploaded dataset:")
    st.dataframe(data.head())

    numeric_data = data.select_dtypes(include=[np.number])
    dropped_cols = list(set(data.columns) - set(numeric_data.columns))
    if dropped_cols:
        st.warning(f"Dropped non-numeric columns: {dropped_cols}")

    data = numeric_data.fillna(0)

    if not np.isfinite(data.values).all():
        st.error("Dataset contains NaN or infinite values. Please clean your data.")
        st.stop()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # ----------- STEP 4: AUTOENCODER MODEL -----------
    input_dim = data_scaled.shape[1]
    encoding_dim = 14

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the model on normal data only (assume first N are normal for demonstration)
    autoencoder.fit(data_scaled[:10000], data_scaled[:10000],
                    epochs=10, batch_size=32, shuffle=True, validation_split=0.2, verbose=0)

    # Reconstruction error
    reconstructed = autoencoder.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 95)
    predictions = (mse > threshold).astype(int)

    # ----------- STEP 5: VISUALIZE RESULTS -----------
    st.subheader("📊 Anomaly Detection Results")
    anomaly_count = np.bincount(predictions)
    st.write(f"✅ Normal: {anomaly_count[0]} | ⚠️ Anomaly: {anomaly_count[1] if len(anomaly_count) > 1 else 0}")

    fig, ax = plt.subplots()
    sns.countplot(x=predictions, ax=ax)
    ax.set_title("Anomaly Count")
    ax.set_xlabel("Prediction (1 = Anomaly)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # ----------- STEP 6: OPTIONAL EVALUATION -----------
    st.subheader("🧪 Optional Evaluation with Labels")
    label_file = st.file_uploader("Upload Label CSV (Optional, must contain 'label' column)", type="csv")
    if label_file is not None:
        try:
            y_true = pd.read_csv(label_file)['label']
            st.text("📄 Classification Report")
            st.text(classification_report(y_true, predictions))
        except Exception as e:
            st.error(f"Could not evaluate labels: {e}")

else:
    st.info("Awaiting CSV file upload...")


