import pennylane as qml
from Initialize_qubits import initialize_qubits

n_qubits = initialize_qubits()  # Number of qubits
dev = qml.device("default.qubit", wires=n_qubits)


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


def initialize_circuit():
    return dev, quantum_circuit_with_memory, n_qubits
