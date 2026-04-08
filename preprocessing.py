import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# NSL-KDD column names
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "outcome","difficulty"
    ]

def load_dataset(path):
    data = pd.read_csv(path, names=columns)
    return data

def preprocess_data(data, scaler=None, train_columns=None):
    # Separate features and label 
    y = data["outcome"].copy() 
    X = data.drop(columns=["outcome","difficulty"])
    # Convert label to binary 
    y = (y != "normal").astype(int) 
    # One-hot encode categorical features 
    X = pd.get_dummies(X, columns=["protocol_type", "service", "flag"]) 
    # Align columns between train/test 
    if train_columns is not None: 
      X = X.reindex(columns=train_columns, fill_value=0) 
    
    # Replace invalid values 
    X = X.replace([np.inf, -np.inf], np.nan) 
    X = X.fillna(0) 
    
    # Normalize 
    if scaler is None: 
      scaler = MinMaxScaler() 
      X_scaled = scaler.fit_transform(X) 
    else:
      X_scaled = scaler.transform(X) 
      
    return X_scaled, y.values, scaler, X.columns


def get_normal_data(X, y):

    normal_mask = (y == 0)

    X_normal = X[normal_mask]

    print("Normal samples found:", len(X_normal))

    if len(X_normal) == 0:
        raise ValueError("No normal samples found in dataset")

    # limit for faster training
    if len(X_normal) > 300000:
        idx = np.random.choice(len(X_normal), 300000, replace=False)
        X_normal = X_normal[idx]

    return X_normal