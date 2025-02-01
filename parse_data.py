import torch
from datasets import load_dataset
import gensim.downloader as api

# Load GloVe 100D embeddings
glove_model = api.load("glove-wiki-gigaword-100")


def text_to_glove(text_list):
    embeddings = []
    for text in text_list:
        words = text.split()
        word_vectors = [
            torch.tensor(glove_model[word]) for word in words if word in glove_model
        ]
        if word_vectors:
            embeddings.append(torch.mean(torch.stack(word_vectors), dim=0))
        else:
            embeddings.append(torch.zeros(100))  # Default vector
    return torch.stack(embeddings)


def get_train_test_data(dataset_name: str):
    dataset = load_dataset(dataset_name)
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    train_X = text_to_glove(train_texts)  # Convert text to embeddings
    test_X = text_to_glove(test_texts)

    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    return train_X.unsqueeze(1), train_y, test_X.unsqueeze(1), test_y
