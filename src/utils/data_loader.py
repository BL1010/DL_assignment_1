import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_dataset(name: str):
    name = name.lower()

    if name == "mnist":
        dataset = fetch_openml("mnist_784", version=1, as_frame=False,parser='liac-arff')

    elif name == "fashion-mnist":
        dataset = fetch_openml("Fashion-MNIST", version=1, as_frame=False,parser = 'liac-arff')

    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion-mnist'")

    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int64)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=10000,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test