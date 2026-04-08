import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Utility function for saving
def get_save_dir(tag):
    save_dir = f"results/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# Reconstruction Error Histogram
def plot_error_distribution(errors, threshold, tag):
    SAVE_DIR = get_save_dir(tag)

    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50)
    plt.axvline(threshold)

    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/reconstruction_error.png")
    plt.show()


# ROC Curve
def plot_roc_curve(y_true, errors, tag):
    SAVE_DIR = get_save_dir(tag)

    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/roc_curve.png")
    plt.show()


# Confusion Matrix
def plot_confusion_matrix(y_true, predictions, tag):
    SAVE_DIR = get_save_dir(tag)

    cm = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/confusion_matrix.png")
    plt.show()


# Error Scatter Plot
def plot_error_scatter(errors, tag):
    SAVE_DIR = get_save_dir(tag)

    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(errors)), errors, s=2)

    plt.title("Reconstruction Error per Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/error_scatter.png")
    plt.show()


# Error Box Plot
def plot_error_box(errors, y_true, tag):
    SAVE_DIR = get_save_dir(tag)

    normal_errors = errors[y_true == 0]
    attack_errors = errors[y_true == 1]

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=[normal_errors, attack_errors])

    plt.xticks([0, 1], ["Normal", "Attack"])
    plt.title("Error Distribution: Normal vs Attack")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/error_box.png")
    plt.show()


# Precision-Recall Curve
def plot_precision_recall(y_true, errors, tag):
    SAVE_DIR = get_save_dir(tag)

    precision, recall, _ = precision_recall_curve(y_true, errors)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/precision_recall.png")
    plt.show()


# Training Loss
def plot_training_loss(history, tag):
    SAVE_DIR = get_save_dir(tag)

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label="Training Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/training_loss.png")
    plt.show()


# Validation Loss
def plot_validation_loss(history, tag):
    SAVE_DIR = get_save_dir(tag)

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['val_loss'], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/validation_loss.png")
    plt.show()


# Error Density Plot
def plot_error_density(errors, tag):
    SAVE_DIR = get_save_dir(tag)

    plt.figure(figsize=(6, 4))
    sns.kdeplot(errors, fill=True)

    plt.title("Reconstruction Error Density")
    plt.xlabel("Error")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/error_density.png")
    plt.show()


# Feature Correlation Heatmap
def plot_feature_correlation(X, tag):
    SAVE_DIR = get_save_dir(tag)

    plt.figure(figsize=(10, 8))
    corr = np.corrcoef(X.T)

    sns.heatmap(corr)

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/feature_correlation.png")
    plt.show()