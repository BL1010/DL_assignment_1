import numpy as np
from utils.data_loader import load_dataset
import argparse
import pickle
from sklearn.metrics import confusion_matrix


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str,
                        default="mnist",
                        choices=["mnist", "fashion-mnist"])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate(model, dataset="mnist"):

    X_train, X_test, y_train, y_test = load_dataset(dataset)

    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    accuracy = np.mean(preds == y_test)

    num_classes = 10
    precision_list = []
    recall_list = []
    f1_list = []

    for c in range(num_classes):

        tp = np.sum((preds == c) & (y_test == c))
        fp = np.sum((preds == c) & (y_test != c))
        fn = np.sum((preds != c) & (y_test == c))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    results = {
        "accuracy": float(accuracy),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1": float(np.mean(f1_list))
    }

    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results


def main():
    args = parse_arguments()
    model = load_model(args.model_path)
    evaluate(model, args.dataset)


if __name__ == "__main__":
    main()