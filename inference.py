import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
from model import CharGPT
from tokenizer import CharTokenizer

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

def predict_next_character(model, context, tokenizer, device='cpu', top_k=3):
    """
    Predicts the next character based on the given context.
    """
    model.to(device)
    model.eval()

    # Debug tokenization
    print(f"\nContext: '{context}'")
    encoded = tokenizer.encode(context, add_special_tokens=False)  # Don't add BOS/EOS for inference
    print(f"Encoded tokens: {encoded}")
    
    # Ensure we have enough context
    if len(encoded) == 0:
        return [("<UNK>", 1.0)]
    
    # Convert context to indices using tokenizer
    input_indices = torch.tensor([encoded], dtype=torch.long, device=device)
    print(f"Input shape: {input_indices.shape}")

    with torch.no_grad():
        logits = model(input_indices)
        last_logits = logits[0, -1, :]
        
        # Debug prediction
        print(f"Logits shape: {logits.shape}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        print(f"Max logit value: {last_logits.max().item():.4f}")
        print(f"Min logit value: {last_logits.min().item():.4f}")

        # Convert logits to probabilities
        probs = torch.softmax(last_logits, dim=0)
        
        # Filter out special tokens
        special_token_ids = set(tokenizer.special_tokens.values())
        for idx in special_token_ids:
            probs[idx] = 0.0
        
        # Renormalize probabilities
        probs = probs / probs.sum()

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(top_k, tokenizer.vocab_size))
        
        # Debug each prediction
        predictions = []
        for idx, prob in zip(top_indices, top_probs):
            token_id = idx.item()
            if token_id in special_token_ids:
                continue
            char = tokenizer.decode([token_id])
            if not char:  # Skip empty strings
                continue
            predictions.append((char, prob.item()))
            print(f"Token {token_id}: '{char}' with probability {prob.item():.4f}")

        # If no valid predictions, return UNK
        if not predictions:
            return [("<UNK>", 1.0)]

    return predictions

def make_vocab():
    tokenizer = CharTokenizer(languages=['en', 'zh', 'ja'])
    return tokenizer.vocab_size, tokenizer.char_to_id, tokenizer.id_to_char

def write_pred(preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

def load_test_data(fname):
    data = []
    with open(fname) as f:
        for line in f:
            inp = line[:-1]  # the last character is a newline
            data.append(inp)
    return data

def predict_test_data(model, test_data):
    vocab_size, char_to_idx, idx_to_char = make_vocab()
    print('\'' in char_to_idx.keys())
    print(char_to_idx)
    for d in test_data:
        pred = predict_next_character(model, d, char_to_idx, idx_to_char)
        print(pred)
        print(pred[0][0])

# ---------------------------
# 3. Running Inference
# ---------------------------
def predict(trained_model_file_path, test_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make vocab  
    vocab_size, char_to_idx, idx_to_char  = make_vocab()
    # load trained model
    model = CharGPT.load_model(
        model_path=trained_model_file_path,
        vocab_size=vocab_size,
        emb_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=100,
        device=device
    )
    # output = generate_text(model, seed, char_to_idx, idx_to_char, device, max_length=200)
    output = predict_next_character(model, test_file_path, len(vocab), char_to_idx, idx_to_char, device=device, top_k=3)

    print("\nGenerated Message:\n", output)

def save_model_and_config(model, tokenizer, save_path):
    """Save both model weights and tokenizer configuration."""
    # Save model weights
    model.save(save_path)
    
    # Save tokenizer config
    config = {
        'vocab_size': tokenizer.vocab_size,
        'char_to_id': tokenizer.char_to_id,
        'id_to_char': tokenizer.id_to_char,
        'special_tokens': tokenizer.special_tokens,
        'languages': tokenizer.languages  # Add this to tokenizer class
    }
    config_path = save_path.replace('.pth', '_tokenizer.pt')
    torch.save(config, config_path)

def load_model_and_config(model_path):
    """Load both model and tokenizer configuration."""
    # Load tokenizer config
    config_path = model_path.replace('.pth', '_tokenizer.pt')
    config = torch.load(config_path)
    
    # Initialize tokenizer with saved config
    tokenizer = PreTrainedCharTokenizer(languages=config['languages'])
    tokenizer.char_to_id = config['char_to_id']
    tokenizer.id_to_char = config['id_to_char']
    tokenizer.special_tokens = config['special_tokens']
    
    # Initialize and load model
    model = CharGPT.load_model(
        model_path=model_path,
        vocab_size=config['vocab_size'],
        emb_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=100
    )
    
    return model, tokenizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load both model and tokenizer
    model, tokenizer = load_model_and_config("charGPT_10.pth")
    model = model.to(device)
    
    print(f"Loaded model vocabulary size: {model.token_embedding.weight.shape[0]}")
    print(f"Loaded tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Test different inputs
    test_inputs = [
        "Simpl",
        "你",
        "The quick brow",
        "世",
    ]

    for seed in test_inputs:
        print(f"\nGenerating predictions for: '{seed}'")
        output = predict_next_character(model, seed, tokenizer, device=device, top_k=3)
        print("\nPredicted next characters:")
        for char, prob in output:
            print(f"'{char}': {prob:.4f}")