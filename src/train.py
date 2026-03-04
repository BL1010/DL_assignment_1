"""
Main Training Script
"""

import argparse
import numpy as np
import wandb
import json
import os
import pickle

from sklearn.model_selection import train_test_split
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default="mnist")
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-o', '--optimizer', type=str,
                        default="adam",
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])

    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers', type=int, default=2)
    parser.add_argument('-sz', '--hidden_size', type=int, default=128)

    parser.add_argument('-a', '--activation', type=str,
                        default="relu",
                        choices=['relu', 'sigmoid', 'tanh'])

    parser.add_argument('-l', '--loss', type=str,
                        default="cross_entropy",
                        choices=['cross_entropy', 'mse'])

    parser.add_argument('-wi', '--weight_init', type=str,
                        default="xavier",
                        choices=['random', 'xavier','zeros'])

    parser.add_argument('--wandb_project', type=str, default="assignment_folder")
    parser.add_argument('--experiment_group', type=str, default="manual_runs")
    parser.add_argument('--model_save_path', type=str, default="models/model.pkl")

    return parser.parse_args()


def compute_accuracy(model, X, y):
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)


def main():
    args = parse_arguments()
    args.hidden_dims = [args.hidden_size] * args.num_layers

    wandb.init(project=args.wandb_project,
               config=vars(args),
               group=args.experiment_group)

    X_train_full, X_test, y_train_full_raw, y_test_raw = load_dataset(args.dataset)

    X_train, X_val, y_train_raw, y_val_raw = train_test_split(
        X_train_full,
        y_train_full_raw,
        test_size=0.1,
        random_state=42,
        stratify=y_train_full_raw
    )

    num_classes = 10

    if args.loss == "cross_entropy":
        y_train = np.eye(num_classes)[y_train_raw]
        y_val = np.eye(num_classes)[y_val_raw]
    else:
        y_train = np.eye(num_classes)[y_train_raw]
        y_val = np.eye(num_classes)[y_val_raw]

    args.input_dim = X_train.shape[1]
    args.output_dim = num_classes

    model = NeuralNetwork(args)

    n_samples = X_train.shape[0]
    best_val_acc = 0.0
    best_model = None
    
    global_step = 0 

    for epoch in range(args.epochs):

        perm = np.random.permutation(n_samples)
        X_train = X_train[perm]
        y_train = y_train[perm]
        y_train_raw = y_train_raw[perm]   # FIXED
        epoch_grad_norm_layer1 = 0 
        num_batches = 0

        for i in range(0, n_samples, args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            y_batch = y_train[i:i+args.batch_size]

            logits = model.forward(X_batch)
            model.backward(y_batch, logits)
            #code snippet for logging of gradient of first hidden layers.
            first_layer = model.layers[0] 
            grad_norm = np.linalg.norm(first_layer.grad_W)
            
            epoch_grad_norm_layer1+= grad_norm
            num_batches+=1
            #-------------------------------------
            # ADDED: Log gradients of 5 neurons 
            #-----------------------------------
            if global_step < 50: 
                wandb.log({
                    "grad_neuron_0": np.mean(first_layer.grad_W[:,0]),
                    "grad_neuron_1": np.mean(first_layer.grad_W[:,1]), 
                    "grad_neuron_2": np.mean(first_layer.grad_W[:,2]),
                    "grad_neuron_3": np.mean(first_layer.grad_W[:,3]),
                    "grad_neuron_4": np.mean(first_layer.grad_W[:,4]), 
                    "iteration": global_step
                })
            model.update_weights()
            global_step+=1

        train_logits = model.forward(X_train)
        val_logits = model.forward(X_val)

        train_loss = model.loss_fn.forward(y_train, train_logits)
        val_loss = model.loss_fn.forward(y_val, val_logits)

        train_acc = compute_accuracy(model, X_train, y_train_raw)
        val_acc = compute_accuracy(model, X_val, y_val_raw)
        
        


        wandb.log({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "grad_norm_layer1": epoch_grad_norm_layer1/num_batches
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = pickle.dumps(model)

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    with open(args.model_save_path, "wb") as f:
        f.write(best_model)

    test_acc = compute_accuracy(model, X_test, y_test_raw)
    wandb.log({"final_test_accuracy": float(test_acc)})

    wandb.finish()


if __name__ == '__main__':
    main()