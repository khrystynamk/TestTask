import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Animal Match Pipeline")
    parser.add_argument('--text', type=str, required=True, help="Text describing the image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the image")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained image classifier model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of the dataset")
    return parser.parse_args()

def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)), # image size for ResNet50
            transforms.ToTensor(), # normalize pizel values to [0, 1]
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # zero-mean, unit-variance
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def check_animal_match(text, image_path, dataset):
    ner_pipeline = pipeline("ner", model="./animal_ner_model", tokenizer="./animal_ner_model")
    entities = ner_pipeline(text)
    detected_animals = set(ent['word'].replace("##", "") for ent in entities if 'LABEL' in ent['entity'])

    model.load_state_dict(torch.load("animal_classifier.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    image_tensor = preprocess_image(image_path).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    predicted_class = dataset.classes[predicted.item()]

    return predicted_class in detected_animals

def run_pipeline(args):
    dataset = datasets.ImageFolder(root=args.data_dir)
    result = check_animal_match(args.text, args.image_path, dataset)
    return result

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
