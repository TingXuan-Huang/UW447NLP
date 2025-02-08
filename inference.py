import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
from model import CharGPT

def load_model(model_path, vocab_size, emb_size=128, num_layers=2, num_heads=4, max_seq_length=100, device='cpu'):
    """
    Loads a trained GPT model from a file.

    Args:
        model_path (str): Path to saved model file.
        vocab_size (int): Number of characters in vocabulary.
        emb_size (int): Embedding size.
        num_layers (int): Number of transformer decoder layers.
        num_heads (int): Number of attention heads per layer.
        max_seq_length (int): Maximum input length.
        device (str): Device ('cpu' or 'cuda').

    Returns:
        model (nn.Module): The loaded model.
    """
    model = CharGPT(vocab_size, emb_size, num_layers, num_heads, max_seq_length)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_text(model, seed_text, char_to_idx, idx_to_char, device='cpu', max_length=100):
    """
    Generates text using a trained GPT model.

    Args:
        model (nn.Module): The trained GPT model.
        seed_text (str): Initial input text.
        char_to_idx (dict): Character to index mapping.
        idx_to_char (dict): Index to character mapping.
        device (str): Device ('cpu' or 'cuda').
        max_length (int): Maximum generated text length.

    Returns:
        generated_text (str): The generated text.
    """
    model.to(device)
    model.eval()
    context = seed_text

    with torch.no_grad():
        for _ in range(max_length):
            input_indices = torch.tensor([[char_to_idx[ch] for ch in context[-50:]]], dtype=torch.long, device=device)
            logits = model(input_indices)
            last_logits = logits[0, -1, :]
            probs = torch.softmax(last_logits, dim=0)
            top_char_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[top_char_idx]
            context += next_char

    return context

def predict_next_character(model, context, char_to_idx, idx_to_char, device='cpu', top_k=3):
    """
    Predicts the next character based on the given context.

    Args:
        model (nn.Module): The trained GPT model.
        context (str): The text context (characters typed so far).
        char_to_idx (dict): Character to index mapping.
        idx_to_char (dict): Index to character mapping.
        device (str): Device ('cpu' or 'cuda').
        top_k (int): Number of top predictions to return.

    Returns:
        predictions (list of tuples): List of (character, probability) predictions.
    """
    model.to(device)
    model.eval()

    # Convert context to indices (use only the last 50 characters)
    input_indices = torch.tensor([[char_to_idx[ch] for ch in context[-50:]]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_indices)  # Forward pass
        last_logits = logits[0, -1, :]  # Get predictions for the last character

        # Convert logits to probabilities
        probs = torch.softmax(last_logits, dim=0)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        predictions = [(idx_to_char[idx.item()], top_probs[i].item()) for i, idx in enumerate(top_indices)]

    return predictions

# ---------------------------
# 3. Running Inference
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocabulary
    vocab = list(string.printable)
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}

    # Load trained model
    model = load_model("gpt_char_model.pth", len(vocab), device=device)

    # Generate text
    seed = "Simpl"
    # output = generate_text(model, seed, char_to_idx, idx_to_char, device, max_length=200)
    output = predict_next_character(model, seed, char_to_idx, idx_to_char, device='cpu', top_k=3)
    print("\nGenerated Message:\n", output)