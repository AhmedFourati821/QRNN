import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR


def train_quantum_rnn(
    model,
    train_data,
    train_labels,
    test_data,
    test_labels,
    epochs=20,
    batch_size=16,
    lr=0.001,
):

    # Loss function for classification
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Create DataLoaders
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs).squeeze(1)
            loss = loss_fn(outputs, batch_targets.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy Calculation
            predicted = (torch.sigmoid(outputs) > 0.5).long()  # Convert logits to 0/1
            correct += (predicted == batch_targets).sum().item()
            total += batch_targets.size(0)
        train_acc = 100 * correct / total
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%"
        )

    # Evaluate on Test Data
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            outputs = model(batch_inputs).squeeze(1)
            predicted = (torch.sigmoid(outputs) > 0.5).long()  # Convert logits to 0/1
            correct += (predicted == batch_targets).sum().item()
            total += batch_targets.size(0)

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    print("Training complete!")
