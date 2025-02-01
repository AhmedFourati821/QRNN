import torch
from torch import nn
import pennylane as qml


class QuantumRecurrentUnit(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        self.quantum_layer = qml.qnn.TorchLayer(
            quantum_circuit, weight_shapes={"weights": (n_layers, n_qubits)}
        )

    def forward(self, x):
        return self.quantum_layer(x)
