import torch
import torch.nn as nn
import torch.optim as optim
import os


def train_model(model, train_loader, test_loader, num_epochs, output_dir, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        train_acc_history.append(train_acc)

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_acc_history.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {
              train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")

    # Збереження моделі
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    return train_acc_history, val_acc_history
