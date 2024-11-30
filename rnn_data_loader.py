import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length

        # Створення словника символів
        self.chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}

        # Перетворення тексту в числові індекси
        self.encoded_text = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        # Вхідна послідовність (X) та вихідна (Y)
        x = self.encoded_text[idx:idx + self.seq_length]
        y = self.encoded_text[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)


def load_text_data(file_path, seq_length=100, batch_size=64):
    # Завантаження тексту
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Створення датасету
    dataset = TextDataset(text, seq_length)

    # Завантажувач даних
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader, dataset
