import torch
from torch import nn
from QuantumRecurrentMemoryUnitClass import QuantumRecurrentUnit
from QuantumFCLayer import quantum_fc_torch


class QuantumRNN(nn.Module):
    def __init__(self, n_qubits, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.qru = QuantumRecurrentUnit(n_qubits, input_dim)
        self.quantum_fc = quantum_fc_torch
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.qru(x)  # Quantum RNN
        x = self.dropout(x)
        results = []
        for i in range(x.shape[0]):
            results.append(
                self.quantum_fc(x[i][0])
            )  # Apply quantum function to single input

        return torch.stack(results)
