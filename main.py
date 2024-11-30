import os
import torch
from data_loader import load_data
from cnn_model import CNNModel
from train_model import train_model
from visualize import save_training_visuals, save_confusion_matrix, save_test_results

# Визначення пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Створення папки для результатів
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# 1. Завантаження даних
train_loader, test_loader = load_data()

# 2. Створення моделі
model = CNNModel(num_classes=10).to(device)

# 3. Навчання моделі
train_acc, val_acc = train_model(
    model, train_loader, test_loader, num_epochs=10, output_dir=output_dir, device=device)

# 4. Оцінка моделі


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


test_acc = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.2f}%")

# 5. Візуалізація результатів
save_training_visuals(train_acc, val_acc, output_dir)
save_confusion_matrix(model, test_loader, device, output_dir)
save_test_results(train_acc, val_acc, output_dir)
