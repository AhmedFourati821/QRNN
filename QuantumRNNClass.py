import torch
from torch import nn
from QuantumRecurrentUnitClass import QuantumRecurrentUnit

class QuantumRNN(nn.Module):
    def __init__(self, n_qubits, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.qru = QuantumRecurrentUnit(n_qubits, input_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.qru(x)  # Pass through quantum recurrent unit

        # Aggregate outputs across time steps (mean pooling)
        x = torch.mean(x, dim=1)

        return self.fc(x)  # Classification layer
