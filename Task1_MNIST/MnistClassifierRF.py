from sklearn.ensemble import RandomForestClassifier
from Task1_MNIST.MnistClassifierInterface import MnistClassifierInterface


class MnistClassifierRF(MnistClassifierInterface):
    def __init__(self, train, test):
        super().__init__(train, test)

        # normalize the data to the range [0, 1]
        self.X_train = train.data.numpy().reshape(len(train), -1) / 255.0
        self.y_train = train.targets.numpy()

        self.X_test = test.data.numpy().reshape(len(test), -1) / 255.0
        self.y_test = test.targets.numpy()

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        if self.model is None:
            raise ValueError("Model is not trained yet. Call `train` first.")
        return self.model.predict(self.X_test)
