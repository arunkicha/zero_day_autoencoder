from train import train_model
from preprocessing import load_dataset, preprocess_data
from evaluate import (
    compute_reconstruction_error,
    compute_threshold,
    detect_anomalies,
    evaluate,
    print_confusion_matrix
)
from visualize import (
    plot_error_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_error_scatter,
    plot_error_box,
    plot_precision_recall,
    plot_error_density,
    plot_feature_correlation,
    plot_training_loss,
    plot_validation_loss
)

import pandas as pd
import os

os.makedirs("results", exist_ok=True)


def run_pipeline(train_path, test_path, use_only_normal=True, tag="normal_only"):

    print(f"\n===== Running Experiment: {tag} =====\n")

    # -------------------------------
    # Train model
    # -------------------------------
    autoencoder, scaler, X_train, history, train_columns, y_train = train_model(
        train_path,
        latent_dim=32,
        use_only_normal=use_only_normal
    )

    # -------------------------------
    # Load & preprocess test data
    # -------------------------------
    print("\nLoading test dataset...\n")

    test_data = load_dataset(test_path)

    X_test, y_test, _, _ = preprocess_data(
        test_data,
        scaler=scaler,
        train_columns=train_columns
    )

    # -------------------------------
    # Compute threshold
    # -------------------------------
    print("\nComputing reconstruction error...\n")

    # ALWAYS use only normal data for threshold
    if use_only_normal:
        train_errors = compute_reconstruction_error(autoencoder, X_train)
    else:
        X_train_normal = X_train[y_train == 0]
        train_errors = compute_reconstruction_error(autoencoder, X_train_normal)

    threshold = compute_threshold(train_errors)

    # -------------------------------
    # Test predictions
    # -------------------------------
    errors = compute_reconstruction_error(autoencoder, X_test)
    predictions = detect_anomalies(errors, threshold)

    print("Threshold:", threshold)

    y_true = y_test

    # -------------------------------
    # Metrics
    # -------------------------------
    precision, recall, f1, auc, accuracy = evaluate(y_true, predictions, errors)

    print("\nEvaluation Results\n")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", auc)

    # Confusion matrix
    cm = print_confusion_matrix(y_true, predictions)

    # -------------------------------
    # Save metrics
    # -------------------------------
    save_dir = f"results/{tag}"
    os.makedirs(save_dir, exist_ok=True)

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Value": [accuracy, precision, recall, f1, auc]
    })

    metrics.to_csv(f"{save_dir}/metrics.csv", index=False)

    # -------------------------------
    # Visualizations (IMPORTANT: pass tag)
    # -------------------------------
    plot_error_distribution(errors, threshold, tag)
    plot_confusion_matrix(y_true, predictions, tag)
    plot_roc_curve(y_true, errors, tag)
    plot_error_scatter(errors, tag)
    plot_error_box(errors, y_true, tag)
    plot_precision_recall(y_true, errors, tag)
    plot_error_density(errors, tag)
    plot_feature_correlation(X_test, tag)
    plot_training_loss(history, tag)
    plot_validation_loss(history, tag)

    print(f"\nResults saved to {save_dir}\n")


# -------------------------------
# Run experiments
# -------------------------------
if __name__ == "__main__":

    # Experiment 1: Normal only
    run_pipeline(
        "data/nsl_kdd/KDDTrain+.txt",
        "data/nsl_kdd/KDDTest+.txt",
        use_only_normal=True,
        tag="normal_only"
    )

    # Experiment 2: Normal + Attack
    run_pipeline(
        "data/nsl_kdd/KDDTrain+.txt",
        "data/nsl_kdd/KDDTest+.txt",
        use_only_normal=False,
        tag="normal_plus_attack"
    )