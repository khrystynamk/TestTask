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


class CNN(nn.Module):
    def __init__(self, num_classes, hidden_dim=32):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=3),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3
            ),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size=3
            ),
            nn.AvgPool2d(2),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.flatten(x)
        return self.fc(x)


class MnistClassifierCNN(MnistClassifierInterface):
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
        self.model = CNN(num_classes, hidden_dim).to(device)

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for _ in tqdm(range(num_epochs)):
            for images, labels in self.train_loader:
                images = images.to(device)
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
