import numpy as np


class CrossEntropy:

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, y_true, logits):

        self.y_true = y_true
        self.probs = self.softmax(logits)

        loss = -np.sum(y_true * np.log(self.probs + 1e-9)) / y_true.shape[0]

        return loss

    def backward(self, y_true, logits):

        probs = self.softmax(logits)

        return (probs - y_true) / y_true.shape[0]


class MSE:

    def forward(self, y_true, y_pred):

        self.y_true = y_true
        self.y_pred = y_pred

        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):

        return 2 * (y_pred - y_true) / y_true.shape[0]