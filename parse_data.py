from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
import torch


def get_train_test_data(dataset_name: str):

    dataset = load_dataset(dataset_name)
    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    vectorizer = CountVectorizer(max_features=n_qubits)  # Limit to n_qubits
    train_X = vectorizer.fit_transform(train_texts).toarray()
    test_X = vectorizer.transform(test_texts).toarray()

    train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(
        1
    )  # Add time dimension
    train_y = torch.tensor(train_labels, dtype=torch.long)
    test_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(1)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    return train_X, train_y, test_X, test_y
