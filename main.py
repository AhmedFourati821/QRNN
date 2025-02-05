from parse_data import get_train_test_data
from train import train_quantum_rnn
from QuantumRNNClass import QuantumRNN
import torch

if __name__ == "__main__":
    # Load dataset
    dataset_name = "imdb"  # Example dataset
    train_X, train_y, test_X, test_y = get_train_test_data(dataset_name)

    # Model Initialization
    n_qubits = 8
    n_layers = 4
    input_dim = 100  # GloVe vectors are 100D
    model = QuantumRNN(n_qubits=n_qubits, input_dim=input_dim, n_layers=n_layers)

    # Train the Quantum RNN
    train_quantum_rnn(
        model, train_X, train_y, test_X, test_y, epochs=30, batch_size=16, lr=0.01
    )
