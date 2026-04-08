import pandas as pd
from preprocessing import preprocess_data
from classical_models import (
run_random_forest,
run_svm,
run_knn,
run_isolation_forest
)

train_path = "/content/drive/MyDrive/zero_day/data/nsl_kdd/KDDTrain+.txt"
test_path = "/content/drive/MyDrive/zero_day/data/nsl_kdd/KDDTest+.txt"

X_train, y_train, scaler, columns = preprocess_data(train_path)
X_test, y_test, _, _ = preprocess_data(test_path, scaler, columns)

print("Train shape:", X_train.shape)
print("Normal samples:", sum(y_train == 0))
print("Attack samples:", sum(y_train == 1))
results = []

rf = run_random_forest(X_train, y_train, X_test, y_test)
results.append(["Random Forest", *rf])

svm = run_svm(X_train, y_train, X_test, y_test)
results.append(["SVM", *svm])

knn = run_knn(X_train, y_train, X_test, y_test)
results.append(["KNN", *knn])

iso = run_isolation_forest(X_train, y_train, X_test, y_test)
results.append(["Isolation Forest", *iso])

df = pd.DataFrame(
results,
columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

print(df)

df.to_csv("results/model_comparison/classical_models.csv", index=False)
