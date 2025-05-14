import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_categorical(df: pd.DataFrame):
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders

def normalize_features(df: pd.DataFrame):
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df, scaler
