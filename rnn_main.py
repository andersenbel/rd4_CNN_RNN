import os
import torch
from rnn_data_loader import load_text_data
from rnn_model import RNNModel
from rnn_train_model import train_rnn, generate_text

# Налаштування
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length = 100
batch_size = 64
num_epochs = 10
lr = 0.001
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Завантаження текстових даних
file_path = 'data/text_corpus.txt'
data_loader, dataset = load_text_data(file_path, seq_length, batch_size)

# Створення моделі
vocab_size = len(dataset.char_to_idx)
model = RNNModel(vocab_size=vocab_size, embedding_dim=128,
                 hidden_dim=256, num_layers=2)

# Тренування моделі
train_rnn(model, data_loader, dataset, num_epochs, lr, device, output_dir)

# Генерація тексту
start_str = "Once upon a time"
gen_text = generate_text(model, dataset, start_str,
                         gen_length=200, device=device)
print(gen_text)

# Збереження тексту
with open(os.path.join(output_dir, 'generated_text.txt'), 'w') as f:
    f.write(gen_text)
