import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_ton_iot(file_path):
    df = pd.read_csv(file_path)
    df = df.select_dtypes(include=["number", "object"]).dropna()

    # Encode categorical features
    for col in df.select_dtypes(include='object'):
        df[col] = LabelEncoder().fit_transform(df[col])
    return df
