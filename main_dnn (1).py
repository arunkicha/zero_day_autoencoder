from sklearn.utils import class_weight
from preprocessing import load_dataset, preprocess_data
from dnn_model import build_dnn

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def run_pipeline(train_path, test_path):

    print("Loading training dataset...")
    train_data = load_dataset(train_path)

    X_train, y_train, scaler = preprocess_data(train_data)

    print("Loading test dataset...")
    test_data = load_dataset(test_path)

    X_test, y_test, _ = preprocess_data(test_data, scaler=scaler)
    input_dim = X_train.shape[1]

    model = build_dnn(input_dim)

    print("Training DNN model...")

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        class_weight={0:1, 1:2}
    )

    print("Predicting test data...")

    preds_prob = model.predict(X_test)
    preds = model.predict(X_test)
    preds = (preds_prob > 0.35).astype(int)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)

    cm = confusion_matrix(y_test, preds)

    print("\nEvaluation Results\n")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", auc)

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":

    run_pipeline(
        "data/nsl_kdd/KDDTrain+.txt",
        "data/nsl_kdd/KDDTest+.txt"
    )