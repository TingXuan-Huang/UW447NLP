import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
from model import CharGPT
from tokenizer import CharTokenizer
from tqdm import tqdm

# ---------------------------
# 1. Create a text dataset.
# ---------------------------
class CharDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=50):
        """
        A simple character-level dataset for next-character prediction.

        Args:
            text (str): The input text corpus.
            tokenizer (CharTokenizer): The tokenizer to use for encoding.
            seq_length (int): The length of input sequences.
        """
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        """
        Returns an (input sequence, target sequence) pair.
        """
        input_text = self.text[idx: idx + self.seq_length]
        target_text = self.text[idx + 1: idx + self.seq_length + 1]  # Shift by 1

        input_indices = torch.tensor(self.tokenizer.encode(input_text), dtype=torch.long)
        target_indices = torch.tensor(self.tokenizer.encode(target_text), dtype=torch.long)

        return input_indices, target_indices

# ---------------------------
# 2. Training Function
# ---------------------------
def train_model(model, dataset, epochs=10, batch_size=16, lr=0.001, device='cpu', accumulation_steps=4):
    """
    Trains the GPT-style model using character-level prediction with gradient accumulation.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.special_tokens["<PAD>"])  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    epoch_pbar = tqdm(range(epochs), desc="Training")
    
    for epoch in epoch_pbar:
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for i, (inputs, targets) in enumerate(batch_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            # Reshape for cross entropy: (batch_size * seq_length, vocab_size)
            outputs = outputs.view(-1, dataset.tokenizer.vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

# ---------------------------
# 3. Running Training
# ---------------------------
if __name__ == "__main__":
    # Set device
    device = torch.device("mps")
    print(f"Using device: {device}")

    # Initialize tokenizer with all required languages
    tokenizer = CharTokenizer(languages=['en', 'zh', 'ja'])
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Create training text that includes all character types
    text_corpus = """
    Hello, World! This is a sample text.
    你好，世界！这是一个示例文本。
    こんにちは世界！これはサンプルテキストです。
    Simple words lead to simple thoughts.
    复杂的词汇导致复杂的思维。
    難しい言葉は難しい考えにつながる。
    """ * 100  # Repeat to create more training data

    # Create dataset
    dataset = CharDataset(text_corpus, tokenizer, seq_length=25)

    # Initialize model with correct vocabulary size
    model = CharGPT(
        vocab_size=tokenizer.vocab_size,
        emb_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=100
    ).to(device)

    # Train model
    print("\nStarting training...")
    train_model(
        model=model,
        dataset=dataset,
        epochs=20,
        batch_size=32,
        lr=0.001,
        device=device
    )

    # Save model
    torch.save(model.state_dict(), "gpt_char_model.pth")
    print("\nModel saved to gpt_char_model.pth")