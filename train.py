import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
from model import CharGPT

# ---------------------------
# 1. Create a text dataset.
# ---------------------------
class CharDataset(Dataset):
    def __init__(self, text, seq_length=50):
        """
        A simple character-level dataset for next-character prediction.

        Args:
            text (str): The input text corpus.
            seq_length (int): The length of input sequences.
        """
        self.text = text
        self.seq_length = seq_length
        self.vocab = list(string.printable)
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        """
        Returns an (input sequence, target sequence) pair.
        """
        input_text = self.text[idx: idx + self.seq_length]
        target_text = self.text[idx + 1: idx + self.seq_length + 1]  # Shift by 1

        input_indices = torch.tensor([self.char_to_idx[ch] for ch in input_text], dtype=torch.long)
        target_indices = torch.tensor([self.char_to_idx[ch] for ch in target_text], dtype=torch.long)

        return input_indices, target_indices

# ---------------------------
# 2. Training Function
# ---------------------------
def train_model(model, dataset, epochs=10, batch_size=64, lr=0.001, device='cpu'):
    """
    Trains the GPT-style model using character-level prediction.

    Args:
        model (nn.Module): The GPT-style transformer model.
        dataset (Dataset): A dataset containing training text.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        device (str): Device ('cpu' or 'cuda').
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # (batch_size, seq_length, vocab_size)
            
            # Reshape for cross-entropy: (batch_size * seq_length, vocab_size)
            outputs = outputs.view(-1, dataset.vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# ---------------------------
# 3. Running Training
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a sample text dataset (Replace this with real training data)
    text_corpus = "This is a sample text corpus to train the character-level GPT model. " * 100
    dataset = CharDataset(text_corpus, seq_length=50)

    # Model Hyperparameters
    emb_size = 128
    num_layers = 2
    num_heads = 4
    max_seq_length = 100

    model = CharGPT(len(dataset.vocab), emb_size, num_layers, num_heads, max_seq_length).to(device)

    # Train the model
    train_model(model, dataset, epochs=10, batch_size=64, lr=0.001, device=device)

    # Save the trained model
    torch.save(model.state_dict(), "gpt_char_model.pth")
    print("Model saved!")