import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def train_qrnn(model, train_X, train_y, test_X, test_y, epochs=10, batch_size=32, learning_rate=0.01):
    # Create DataLoader for training data
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            optimizer.step()

            total_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate the model on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_X)
        _, predictions = torch.max(test_outputs, dim=1)
        accuracy = (predictions == test_y).float().mean()
        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")