import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_train, y_train, X_test, y_test):

  model.fit(X_train, y_train)
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  precision = precision_score(y_test, predictions)
  recall = recall_score(y_test, predictions)
  f1 = f1_score(y_test, predictions)

  return accuracy, precision, recall, f1

def run_random_forest(X_train, y_train, X_test, y_test):

  model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
  )

  return evaluate_model(model, X_train, y_train, X_test, y_test)


def run_svm(X_train, y_train, X_test, y_test):

  model = SVC(
    kernel="rbf",
    gamma="scale",
    class_weight="balanced"
  )

  return evaluate_model(model, X_train, y_train, X_test, y_test)

def run_knn(X_train, y_train, X_test, y_test):


  model = KNeighborsClassifier(
    n_neighbors=5
  )

  return evaluate_model(model, X_train, y_train, X_test, y_test)

def run_isolation_forest(X_train, y_train, X_test, y_test):

  model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
  )

  model.fit(X_train)

  predictions = model.predict(X_test)

  predictions = np.where(predictions == -1, 1, 0)

  accuracy = accuracy_score(y_test, predictions)
  precision = precision_score(y_test, predictions)
  recall = recall_score(y_test, predictions)
  f1 = f1_score(y_test, predictions)
  return accuracy, precision, recall, f1

