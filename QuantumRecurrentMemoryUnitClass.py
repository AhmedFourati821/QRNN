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
        batch_size, time_steps, _ = x.shape
        quantum_outputs = []

        for t in range(time_steps):
            truncated_input = x[:, t, : self.n_qubits]
            batch_outputs = torch.stack(
                [
                    self.quantum_layer(truncated_input[i])
                    for i in range(truncated_input.shape[0])
                ]
            )
            quantum_outputs.append(batch_outputs)

        # Stack outputs across time steps
        return torch.stack(quantum_outputs, dim=1)
