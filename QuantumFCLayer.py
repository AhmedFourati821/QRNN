import pennylane as qml
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