import torch
from torch import nn
from QuantumRecurrentUnitClass import QuantumRecurrentUnit
import pennylane as qml


class QuantumRNN(nn.Module):
    def __init__(self, n_qubits, input_dim, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.qru = QuantumRecurrentUnit(n_qubits, n_layers)

        self.clayer = torch.nn.Linear(input_dim, n_qubits)
        self.output_layer = torch.nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = self.clayer(x)

        x_split = torch.split(x, self.input_dim // self.n_qubits, dim=1)
        x_split = list(x_split)  # Convert tuple to list

        for i in range(len(x_split)):
            x_split[i] = self.qru(x_split[i])

        x = torch.cat(x_split, axis=1)
        return self.output_layer(x).squeeze(1)
