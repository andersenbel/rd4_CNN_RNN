import torch
import torch.nn as nn
import os


def train_rnn(model, data_loader, dataset, num_epochs, lr, device, output_dir):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Ініціалізація прихованого стану з урахуванням batch_size
            batch_size = inputs.size(0)
            hidden = (torch.zeros(2, batch_size, 256).to(device),
                      torch.zeros(2, batch_size, 256).to(device))

            # Передбачення
            outputs, hidden = model(inputs, hidden)

            # Обчислення втрат
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), targets.view(-1))

            # Зворотне поширення
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Збереження моделі
    torch.save(model.state_dict(), os.path.join(output_dir, 'rnn_model.pth'))
    return dataset


def generate_text(model, dataset, start_str, gen_length, device):
    model.eval()
    input_seq = [dataset.char_to_idx[ch] for ch in start_str]
    input_seq = torch.tensor(
        input_seq, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_str
    hidden = None

    for _ in range(gen_length):
        with torch.no_grad():
            outputs, hidden = model(input_seq, hidden)
            next_char_idx = outputs[0, -1].argmax().item()
            next_char = dataset.idx_to_char[next_char_idx]
            generated_text += next_char

            input_seq = torch.tensor(
                [[next_char_idx]], dtype=torch.long).to(device)

    return generated_text
