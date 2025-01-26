import torch
from torch import nn
from quantum_cirucit import quantum_circuit

class QuantumRecurrentUnit(nn.Module):
    def __init__(self, n_qubits, input_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.weight = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x):
        quantum_outputs = []
        for t in range(x.shape[1]):  # Iterate over time steps
            truncated_input = x[:, t, :self.n_qubits]

            # Process each sample in the batch
            batch_outputs = []
            for inp in truncated_input:
                quantum_out = quantum_circuit(inp, self.weight)  # Process one sample

                # If quantum_circuit returns a list, convert it to a tensor
                if isinstance(quantum_out, list):
                    quantum_out = torch.tensor(quantum_out, dtype=torch.float32)

                batch_outputs.append(quantum_out)

            # Stack all outputs in the batch
            batch_outputs = torch.stack(batch_outputs)
            quantum_outputs.append(batch_outputs)

        # Stack outputs across time steps
        return torch.stack(quantum_outputs, dim=1)