import numpy as np
from preprocessing import load_dataset, preprocess_data, get_normal_data
from autoencoder_model import build_autoencoder
from tensorflow.keras.callbacks import EarlyStopping


def train_model(train_path, latent_dim=24, use_only_normal=True):
    print("Loading dataset...")
    data = load_dataset(train_path)

    print("Preprocessing dataset...")
    X, y, scaler, train_columns = preprocess_data(data)

    if use_only_normal:
        print("Using only normal data...")
        normal_mask = (y == 0)

        X_train = X[normal_mask]
        y_train_filtered = y[normal_mask]

        print("Normal samples found:", len(X_train))

        if len(X_train) == 0:
            raise ValueError("No normal samples found in dataset")

        # limit for faster training
        if len(X_train) > 300000:
            idx = np.random.choice(len(X_train), 300000, replace=False)
            X_train = X_train[idx]
            y_train_filtered = y_train_filtered[idx]

    else:
        print("Using full dataset (normal + attack)...")
        X_train = X
        y_train_filtered = y

    print("Training samples:", X_train.shape)

    # model
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim, latent_dim)

    print("Training autoencoder...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    )

    # Train model
    history = autoencoder.fit(
        X_train,
        X_train,
        epochs=150,
        batch_size=64,
        validation_split=0.2,
        shuffle=True,
        callbacks=[early_stop]
    )

    print("Training completed")
    return autoencoder, scaler, X_train, history, train_columns, y_train_filtered