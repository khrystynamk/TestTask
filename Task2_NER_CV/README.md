# Task2_NER_CV

This folder contains the code and files necessary for a task related to animal classification using both image classification and Named Entity Recognition (NER). It implements a pipeline where the user provides both a text description and an image, and the system predicts whether the text matches the animal in the image.

## File Structure

### `ner_dataset_generation.py`
This script is responsible for generating the data used for training the Named Entity Recognition (NER) model. It processes raw text data to identify and label animal entities that can be used in training a NER model.

### `animal10_eda.ipynb`
This Notebook contains exploratory data analysis (EDA) on the `Animals-10` dataset. It provides visualizations and summary statistics to understand the dataset better before training the image classification model.

### `train_image_clf.py`
This script trains the image classification model on the `Animals-10` dataset using a pre-trained model (ResNet50). It performs fine-tuning and saves the trained model weights for future use.

Example command:

```bash
python train_image_clf.py --data_dir "./Animals-10" --output_model "animal_classifier.pth" --batch_size 32 --num_epochs 5 --learning_rate 0.001
```

### `infer_image_clf.py`
This script is used for making inferences using a pre-trained image classification model. The user provides an image, and the model predicts the animal class in the image.

Example command:

```bash
python infer_image_clf.py --model_path "animal_classifier.pth" --image_path "./Anomals-10/chicken/chicken (1).jpeg" --data_dir "./Animals-10"
```

### `train_ner.py`
This script is used for training the NER model. It takes a dataset of text and labels, processes it, and trains a model to identify animal names from the text.

Example command:

```bash
python train_ner.py --data_path "animal_ner_dataset.json" --model_name "distilbert-base-cased" --output_dir "./animal_ner_model" --batch_size 16 --num_epochs 3 --lr 2e-5
```

### `infer_ner.py`
This script performs Named Entity Recognition (NER) on input text. The user provides a text description, and the model detects any animal names in the text.

Example command:

```bash
python infer_ner.py --model_dir "./animal_ner_model" --text "There is a cow in the picture."
```

### `match_pipeline.py`
This script is the main entry point for the animal match pipeline. It combines both the image classification and NER models to check if the text description of an animal matches the animal in the image. The user provides an image and text, and the script returns a boolean indicating whether the animal in the image matches the text description.

Example command:

```bash
python animal_match_pipeline.py \
    --text "There is a cat in the image." \
    --image_path "./Animals-10/cat/cat (1).jpeg" \
    --model_path "animal_classifier.pth" \
    --data_dir "Animals-10"
```

## Installation

Before launching any scripts, it is necessary to have all the data:

- The dataset for NER training can be generated after running `ner_dataset_generation.py` script.
- The Animal-10 image dataset can be downloaded from `kaggle` in the `animal10_eda.ipynb`.

To install the dependencies, use the following command:

```bash
pip install -r requirements.txt
```
