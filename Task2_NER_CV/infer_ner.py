import argparse
from transformers import pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for Named Entity Recognition (NER)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the trained model is saved",
    )
    parser.add_argument("--text", type=str, required=True, help="Input text for NER")
    return parser.parse_args()


def infer_ner(args):
    ner_pipeline = pipeline("ner", model=args.model_dir, tokenizer=args.model_dir)

    result = ner_pipeline(args.text)
    print(result)


if __name__ == "__main__":
    args = parse_args()
    infer_ner(args)
