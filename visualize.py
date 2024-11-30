import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def save_training_visuals(train_acc, val_acc, output_dir):
    """
    Зберігає графік точності для навчальної та валідаційної вибірок.
    """
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig(os.path.join(output_dir, 'history.png'))
    plt.close()


def save_confusion_matrix(model, test_loader, device, output_dir):
    """
    Зберігає матрицю плутанини як зображення.
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Проходимо тестовий датасет
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Обчислення матриці плутанини
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.arange(10))

    # Побудова матриці
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


def save_test_results(train_acc, val_acc, output_dir):
    """
    Зберігає результати навчання та валідації у текстовий файл.
    """
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Final Train Accuracy: {train_acc[-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_acc[-1]:.2f}%\n")
