import pennylane as qml
from pennylane import numpy as np

n_qubits = 4  # Number of qubits
dev = qml.device("default.qubit", wires=n_qubits)


# Old quantum circuit without quantum recurrence
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(min(len(inputs), n_qubits)):  # Use only valid qubits
        qml.RX(inputs[i], wires=i)
    for i in range(n_qubits - 1):
        qml.RY(weights[i], wires=i)
        qml.CZ(wires=[i, i + 1])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
