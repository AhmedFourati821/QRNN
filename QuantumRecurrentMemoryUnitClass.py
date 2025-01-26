from quantum_cirucit import quantum_circuit_with_memory
import torch
from torch import nn

class QuantumRecurrentUnit(nn.Module):
    def __init__(self, n_qubits, input_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.weight = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x):
        batch_size, time_steps, _ = x.shape
        prev_state = torch.zeros(batch_size, self.n_qubits, dtype=torch.float32)  # Initialize quantum memory
        quantum_outputs = []

        for t in range(time_steps):  # Iterate over time steps
            truncated_input = x[:, t, :self.n_qubits]  # Limit input features to number of qubits

            # Process each sample in the batch
            batch_outputs = []
            for b in range(batch_size):
                quantum_out = quantum_circuit_with_memory(prev_state[b], truncated_input[b], self.weight)
                quantum_out = torch.tensor(quantum_out, dtype=torch.float32)  # Convert to tensor
                batch_outputs.append(quantum_out)

            # Stack batch outputs and update the quantum memory
            batch_outputs = torch.stack(batch_outputs)
            prev_state = batch_outputs  # Update quantum memory
            quantum_outputs.append(batch_outputs)

        # Stack outputs across time steps
        return torch.stack(quantum_outputs, dim=1)