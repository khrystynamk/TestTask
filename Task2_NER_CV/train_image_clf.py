import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Image Classification model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory of the dataset"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="animal_classifier.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    return parser.parse_args()


def train_image_classifier(args):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # image size for ResNet50
            transforms.ToTensor(),  # normalize pixel values to [0, 1]
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # zero-mean, unit-variance
        ]
    )

    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(dataset.classes)) # adjust number of output labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss / len(train_loader):.4f}"
        )

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == "__main__":
    args = parse_args()
    train_image_classifier(args)
