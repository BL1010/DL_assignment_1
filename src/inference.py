import numpy as np
import argparse
import wandb

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

    # get weight keys in order
    W_keys = sorted([k for k in weights.keys() if k.startswith("W")],
                    key=lambda x: int(x[1:]))

    input_dim = weights[W_keys[0]].shape[0]
    output_dim = weights[W_keys[-1]].shape[1]

    hidden_dims = []
    for i in range(len(W_keys) - 1):
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

    # convert one-hot labels to class indices if needed
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    logits = model.forward(X_test)

    # argmax of logits is same as argmax of softmax
    preds = np.argmax(logits, axis=1)

    accuracy = np.mean(preds == y_test)

    num_classes = len(np.unique(y_test))
    f1_scores = []

    for c in range(num_classes):

        tp = np.sum((preds == c) & (y_test == c))
        fp = np.sum((preds == c) & (y_test != c))
        fn = np.sum((preds != c) & (y_test == c))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1_scores.append(f1)

    metrics =  {
        "accuracy": float(accuracy),
        "f1": float(np.mean(f1_scores))
    }
    return metrics, preds, y_test, X_test


def main():

    args = parse_arguments()
    wandb.init(project="mnist-error-analysis",config=vars(args))

    model = load_model(args.model_path)

    metrics, preds, y_test, X_test = evaluate(model, args.dataset)

    print(metrics)
    wandb.log(metrics) 
    
    #confusion matrix 
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs = None, 
            y_true = y_test , 
            preds = preds, 
            class_names=[str(i) for i in range(10)]
        )
    })
    
    #creative visualization 
    mis_idx = np.where(preds!=y_test)[0][:20] 
    
    mis_images = [] 
    for i in mis_idx: 
        img = X_test[i].reshape(28,28) 
        mis_images.append(
            wandb.Image(img,caption = f"True:{y_test[i]} Pred:{preds[i]}")
            
        )
    wandb.log({"Misclassified_examples": mis_images})
    


if __name__ == "__main__":
    main()