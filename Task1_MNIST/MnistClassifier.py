import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Task1_MNIST.MnistClassifierRF import MnistClassifierRF
from Task1_MNIST.MnistClassifierNN import MnistClassifierNN
from Task1_MNIST.MnistClassifierCNN import MnistClassifierCNN


class MnistClassifier:
    def __init__(self, algorithm, train, test):
        if algorithm == "rf":
            self.model = MnistClassifierRF(train, test)
        elif algorithm == "nn":
            self.model = MnistClassifierNN(train, test)
        elif algorithm == "cnn":
            self.model = MnistClassifierCNN(train, test)
        else:
            raise ValueError(f"Unsupported algorithm has been provided: {algorithm}")

    def train(self):
        self.model.train()

    def predict(self):
        y_pred = self.model.predict()
        y_true = (
            self.model.y_test
            if hasattr(self.model, "y_test")
            else self.model.test_loader.dataset.targets.numpy()
        )

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        return pd.DataFrame(
            [
                {
                    "y_pred": y_pred,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            ]
        )
