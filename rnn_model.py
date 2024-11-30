import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(RNNModel, self).__init__()
        # Вбудовування символів
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Рекурентний шар (LSTM)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers, batch_first=True)

        # Вихідний шар для передбачення ймовірностей
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
