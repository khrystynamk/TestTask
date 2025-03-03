import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from transformers import Trainer
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Named Entity Recognition (NER) model"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset file (JSON)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-cased",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ner_model",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of epochs for training"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    return parser.parse_args()


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []

        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(label[word_id])
            prev_word_id = word_id

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def train_ner(args):
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    train_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        save_total_limit=2,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    train_ner(args)
