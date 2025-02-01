from quantum_cirucit import initialize_circuit
import torch
from torch import nn
import pennylane as qml


class QuantumRecurrentUnit(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        dev = qml.device("lightning.qubit", wires=n_qubits)

        # Adding quantum memory to propagate information across time steps
        @qml.qnode(dev, interface="torch")
        def quantum_circuit_with_memory(inputs, weights):
            # Encode the input into the quantum system
            for i in range(len(inputs)):
                qml.RY(inputs[i], wires=i)

            # Apply parameterized quantum gates
            for _ in range(6):
                for i in range(len(weights)):
                    qml.RY(weights[i], wires=i)
                    qml.CZ(wires=[i, (i + 1) % n_qubits])  # Entanglement

            # Measure qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_layer = qml.qnn.TorchLayer(
            qml.QNode(quantum_circuit_with_memory, dev, interface="torch"),
            weight_shapes={"weights": (n_qubits,)},
        )

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
