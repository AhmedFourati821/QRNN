import pennylane as qml
from pennylane import numpy as np
from Initialize_qubits import initialize_qubits

n_qubits = initialize_qubits()  # Number of qubits
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

#Adding quantum memory to propagate information across time steps
@qml.qnode(dev, interface="torch")
def quantum_circuit_with_memory(prev_state, inputs, weights):
    # Encode the previous state into the quantum system
    for i in range(len(prev_state)):
        qml.RX(prev_state[i], wires=i)
    # Encode the current input into the quantum system
    for i in range(len(inputs)):
        qml.RY(inputs[i], wires=i)
    # Apply entanglement and parameterized gates
    for i in range(len(weights)):
        qml.RY(weights[i], wires=i)
        qml.CZ(wires=[i, (i + 1) % n_qubits])
    # Return the updated state
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]