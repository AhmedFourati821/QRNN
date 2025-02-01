import torch
from torch import nn
from QuantumRecurrentMemoryUnitClass import QuantumRecurrentUnit
from QuantumFCLayer import quantum_fc_torch
import pennylane as qml

class QuantumRNN(nn.Module):
    def __init__(self, n_qubits, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.qru = QuantumRecurrentUnit(n_qubits, input_dim)
        dev_fc = qml.device("lightning.qubit", wires=n_qubits)
        @qml.qnode(dev_fc, interface="torch")
        def quantum_fc_layer(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)  

            for i in range(n_qubits):
                qml.RY(weights[i], wires=i)
                qml.CZ(wires=[i, (i + 1) % n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        quantum_fc_torch = qml.qnn.TorchLayer(
            quantum_fc_layer, weight_shapes={"weights": (n_qubits,)}
        )
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
