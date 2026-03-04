import numpy as np
from utils.data_loader import load_dataset
import argparse
import pickle
from argparse import Namespace
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference")

    parser.add_argument("--model_path", type=str, default=None)
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

    if model_path.endswith(".npy"):

        weights = np.load(model_path, allow_pickle=True)

        # handle 0-d array
        if weights.shape == ():
            weights = weights.item()

        # if dict
        if isinstance(weights, dict):
            weights = list(weights.values())

        # if object array
        if isinstance(weights, np.ndarray):
            weights = list(weights)

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
            return pickle.load(f)


def evaluate(model, dataset="mnist"):

    _, X_test, _, y_test = load_dataset(dataset)

    logits = model.forward(X_test)
    preds = np.argmax(logits, axis=1)

    accuracy = np.mean(preds == y_test)

    return {"accuracy": float(accuracy)}


def main():
    args = parse_arguments()
    if args.model_path is None:
        return
    model = load_model(args.model_path)
    evaluate(model, args.dataset)


if __name__ == "__main__":
    main()