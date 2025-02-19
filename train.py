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
    def __init__(self, text, tokenizer, seq_length=25):
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
        return len(self.text) - self.seq_length + 1

    def __getitem__(self, idx):
        """
        Returns an (input sequence, target sequence) pair.
        """
        # Get the input and target text
        input_text = self.text[idx: idx + self.seq_length]
        
        # Ensure the target text does not exceed the length of the text
        if idx + self.seq_length < len(self.text):
            target_text = self.text[idx + 1: idx + self.seq_length + 1]  # Shift by 1
        else:
            target_text = self.text[idx + 1:]  # Take the remaining text if at the end

        # Encode the input and target texts without special tokens
        input_indices = torch.tensor(self.tokenizer.encode(input_text), dtype=torch.long)
        target_indices = torch.tensor(self.tokenizer.encode(target_text), dtype=torch.long)

        # Add <BOS> to the beginning of input_indices
        input_indices = torch.cat((torch.tensor([self.tokenizer.special_tokens["<BOS>"]], dtype=torch.long), input_indices))

        # Add <EOS> to the end of target_indices
        target_indices = torch.cat((target_indices, torch.tensor([self.tokenizer.special_tokens["<EOS>"]], dtype=torch.long)))

        # Adjust lengths after adding special tokens
        input_length = len(input_indices)
        target_length = len(target_indices)

        # If the lengths exceed seq_length, truncate them
        if input_length > self.seq_length:
            input_indices = input_indices[:self.seq_length]
        if target_length > self.seq_length:
            target_indices = target_indices[:self.seq_length]

        # After decoding
        decoded_input = self.tokenizer.decode(input_indices)
        decoded_target = self.tokenizer.decode(target_indices)

        return input_indices, target_indices

# ---------------------------
# 2. Training Function
# ---------------------------
def train_model(model, dataset, epochs=10, batch_size=16, 
                lr=0.001, device='cpu', accumulation_steps=4, save_model = True):
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
        
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            outputs = outputs.view(-1, dataset.tokenizer.vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            # Update epoch progress bar with current loss
            epoch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

        if save_model and epoch % 10 == 0:
            model.save(f"charGPT_{epoch}.pth")

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
    """  # Repeat to create more training data

    # Create dataset
    dataset = CharDataset(text_corpus, tokenizer, seq_length=25)

    # Initialize model with correct vocabulary size
    model = CharGPT(
        vocab_size=tokenizer.vocab_size,
        emb_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=25
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