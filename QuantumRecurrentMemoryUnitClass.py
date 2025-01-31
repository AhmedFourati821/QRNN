from quantum_cirucit import initialize_circuit
import torch
from torch import nn
import pennylane as qml

dev, quantum_circuit_with_memory, n_qubits = initialize_circuit()

quantum_layer = qml.qnn.TorchLayer(
    qml.QNode(quantum_circuit_with_memory, dev, interface="torch"),
    weight_shapes={"weights": (n_qubits,)},
)


class QuantumRecurrentUnit(nn.Module):
    def __init__(self, n_qubits, input_dim):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum_layer = quantum_layer

    def forward(self, x):
        quantum_outputs = []

        for t in range(x.shape[1]):  # Iterate over time steps
            truncated_input = x[:, t, : self.n_qubits]  # Ensure correct input shape

            # Process each sample in the batch individually
            batch_outputs = torch.stack(
                [
                    self.quantum_layer(truncated_input[i])
                    for i in range(truncated_input.shape[0])
                ]
            )

            quantum_outputs.append(batch_outputs)

        return torch.stack(quantum_outputs, dim=1)
