import numpy as np
import argparse

from argparse import Namespace
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="src/best_model.npy")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--weight_init", type=str, default="xavier")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)

    return parser.parse_args()


def load_model(model_path):

    weights = np.load(model_path, allow_pickle=True)

    if isinstance(weights, np.ndarray):
        weights = weights.item()
    W_keys = sorted([k for k in weights.keys() if k.startswith("W")])
    input_dim = weights[W_keys[0]].shape[0]
    output_dim = weights[W_keys[-1]].shape[1]

    hidden_dims = []
    for i in range(len(W_keys)-1):
        hidden_dims.append(weights[f"W{i}"].shape[1])

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
    model.set_weights(weights)

    return model


def evaluate(model, dataset="mnist"):

    _, X_test, _, y_test = load_dataset(dataset)

    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    accuracy = np.mean(preds == y_test)

    num_classes = 10
    f1_scores = []

    for c in range(num_classes):

        tp = np.sum((preds == c) & (y_test == c))
        fp = np.sum((preds == c) & (y_test != c))
        fn = np.sum((preds != c) & (y_test == c))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_scores.append(f1)

    return {
        "accuracy": float(accuracy),
        "f1": float(np.mean(f1_scores))
    }


def main():

    args = parse_arguments()

    model = load_model(args.model_path)

    metrics = evaluate(model, args.dataset)

    print(metrics)


if __name__ == "__main__":
    main()