import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn

from tqdm import tqdm
from Task1_MNIST.MnistClassifierInterface import MnistClassifierInterface

batch_size = 64
input_size = 784  # 28 * 28, size of the image
hidden_dim = 256
num_classes = 10
learning_rate = 0.001
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.layers(x)
        return logits


class MnistClassifierNN(MnistClassifierInterface):
    def __init__(self, train, test):
        super().__init__(train, test)

        # converts a PIL Image or ndarray to tensor and scales the values to [0, 1]
        train.transform = transforms.ToTensor()
        test.transform = transforms.ToTensor()

        self.train_loader = torch.utils.data.DataLoader(
            dataset=train, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test, batch_size=batch_size, shuffle=False
        )
        self.model = FeedForward(input_size, hidden_dim, num_classes).to(device)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for _ in tqdm(range(num_epochs)):
            for images, labels in self.train_loader:
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions
