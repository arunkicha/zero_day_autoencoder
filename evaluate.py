import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

def compute_reconstruction_error(model, X):

  reconstructions = model.predict(X)
  mse = np.mean((X - reconstructions) ** 2, axis=1)
  return mse

def compute_threshold(train_errors):

  threshold = np.mean(train_errors) + 3 * np.std(train_errors)
  return threshold

def detect_anomalies(errors, threshold):

  predictions = (errors > threshold).astype(int)
  return predictions

def evaluate(y_true, predictions, errors):

  precision = precision_score(y_true, predictions)
  recall = recall_score(y_true, predictions)
  f1 = f1_score(y_true, predictions)
  auc = roc_auc_score(y_true, errors)
  accuracy = accuracy_score(y_true, predictions)

  return precision, recall, f1, auc, accuracy


def print_confusion_matrix(y_true, predictions):

  cm = confusion_matrix(y_true, predictions)

  print("Confusion Matrix:")
  print(cm)

  return cm

