import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for Image Classification model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the image for classification",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory of the dataset",
    )
    return parser.parse_args()


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # image size for ResNet50
            transforms.ToTensor(),  # normalize pixel values to [0, 1]
            transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),  # zero-mean, unit-variance
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def infer_image_classifier(args):
    dataset = datasets.ImageFolder(root=args.data_dir)
    class_names = dataset.classes

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    image_tensor = preprocess_image(args.image_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    print(f"Predicted class: {predicted_class}")


if __name__ == "__main__":
    args = parse_args()
    infer_image_classifier(args)
