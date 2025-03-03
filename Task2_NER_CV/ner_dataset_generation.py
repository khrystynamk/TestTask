import json
from transformers import AutoTokenizer

# animal entities from the Animal-10 dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
animals = [
    "dog",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]

templates = [
    "There is a {animal} in the picture.",
    "A {animal} appears in the image.",
    "This photo contains a {animal}.",
    "You can see a {animal} in the picture.",
    "A {animal} is present in this image.",
    "The picture shows a {animal}.",
    "A {animal} can be found in this photo.",
    "There’s definitely a {animal} in this image.",
    "The image includes a {animal}.",
    "I can spot a {animal} in this photo.",
    "Is that a {animal} in the picture?",
    "Can you see a {animal} in the image?",
    "Does this photo contain a {animal}?",
    "Could that be a {animal} in this image?",
    "Looks like there’s a {animal} in this photo.",
    "I think I see a {animal} in this picture.",
    "This might be a {animal} in the photo.",
    "It appears that a {animal} is present in this image.",
    "I’d say this is a picture of a {animal}.",
    "A {animal} is visible in the photo.",
    "A {animal} is right in the center of this image.",
    "This shot captures a {animal} looking toward the camera.",
    "The photo is focused on a {animal} that seems to be eating.",
    "The picture features a {animal} playing with another animal.",
    "There is a {animal} in the center of the field.",
    "A {animal} is standing near the riverbank.",
    "You can see a {animal} resting under a tree.",
    "A {animal} is walking through the forest.",
    "This image shows a {animal} running in the open grassland.",
    "The {animal} in the photo is drinking water from a pond.",
    "A {animal} is hiding behind the bushes.",
    "A {animal} is climbing a tree in this picture.",
    "The {animal} appears to be hunting its prey in the wild.",
    "There’s a {animal} resting on a rock in this image.",
    "A {animal} is playing with its cubs in the background.",
    "A {animal} is lying in the shade to avoid the heat.",
    "The {animal} is running along the river in this scene.",
    "This photo captures a {animal} gazing into the distance.",
    "A {animal} is seen walking along the beach.",
    "The {animal} in this picture seems to be chasing something.",
    "A {animal} is curled up, taking a nap in this photo.",
    "You can spot a {animal} near the water’s edge.",
    "A {animal} is leaping through the tall grass in this image.",
    "The image depicts a {animal} interacting with its surroundings.",
    "In this picture, a {animal} appears to be exploring its habitat.",
    "The {animal} is perched on a rock surveying the area.",
    "A {animal} is peacefully grazing in the open field.",
    "This snapshot features a {animal} moving through the dense forest.",
    "The {animal} in the image is looking curiously at the camera.",
    "A {animal} is splashing around in the water.",
    "This image shows a {animal} stretching in the early morning light.",
    "A {animal} is playing with a group of other animals in this photo.",
    "You can see a {animal} standing majestically on the hilltop.",
    "This picture highlights a {animal} resting after a long journey.",
]


def generate_ner_data():
    dataset = []

    for animal in animals:
        for template in templates:
            sentence = template.format(animal=animal)

            tokens = tokenizer.tokenize(sentence)
            tokenized_sentence = tokenizer.convert_tokens_to_string(tokens).split()

            label_ids = [0] * len(tokenized_sentence)  # default, 0 = "O"
            words = sentence.split()

            for i, word in enumerate(words):
                if animal in word:
                    label_ids[i] = 1  # 1 = "ANIMAL"

            dataset.append({"tokens": tokenized_sentence, "ner_tags": label_ids})

    return dataset


if __name__ == "__main__":
    ner_data = generate_ner_data()

    with open("animal_ner_dataset.json", "w", encoding="utf-8") as f:
        json.dump(ner_data, f, indent=4, ensure_ascii=False)

    print("Dataset generated and saved as animal_ner_dataset.json")
