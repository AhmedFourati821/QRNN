from parse_data import get_train_test_data
from train import train_qrnn
from QRNN.QuantumRNNClass import QuantumRNN
import torch
from Initialize_qubits import initialize_qubits

if __name__ == "__main__":
    # Load dataset
    dataset_name = "imdb"  # Example dataset
    train_X, train_y, test_X, test_y = get_train_test_data(dataset_name)

    # Initialize the model
    n_qubits = initialize_qubits()
    input_dim = train_X.shape[1]
    hidden_dim = 4
    output_dim = len(torch.unique(train_y))
    model = QuantumRNN(n_qubits, input_dim, hidden_dim, output_dim)

    # Train the model
    train_qrnn(model, train_X, train_y, test_X, test_y, epochs=10, batch_size=32, learning_rate=0.01)