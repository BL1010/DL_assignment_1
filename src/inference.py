import numpy as np
from utils.data_loader import load_dataset
import argparse
import pickle
from argparse import Namespace
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str,
                        default="mnist",
                        choices=["mnist", "fashion-mnist"])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_model(model_path):

    if model_path is None:
        return None

    if model_path.endswith(".npy"):

        weights = np.load(model_path, allow_pickle=True)

        if isinstance(weights, np.ndarray) and weights.shape == ():
            weights = weights.item()

        input_dim = weights[0].shape[0]
        output_dim = weights[-2].shape[1]

        hidden_dims = []
        for i in range(0, len(weights) - 2, 2):
            hidden_dims.append(weights[i].shape[1])

        args = Namespace(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            loss="cross_entropy",
            optimizer="sgd",
            learning_rate=0.01,
            weight_decay=0.0,
            weight_init="xavier"
        )

        model = NeuralNetwork(args)

        layer_idx = 0
        for i in range(0, len(weights), 2):
            model.layers[layer_idx].W = weights[i]
            model.layers[layer_idx].b = weights[i + 1]
            layer_idx += 1

        return model

    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model


def evaluate(model, dataset="mnist"):

    _, X_test, _, y_test = load_dataset(dataset)

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

    return {
        "accuracy": float(accuracy),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "f1": float(np.mean(f1_list))
    }


def main():
    args = parse_arguments()
    if args.model_path is None:
        return
    model = load_model(args.model_path)
    evaluate(model, args.dataset)


if __name__ == "__main__":
    main()