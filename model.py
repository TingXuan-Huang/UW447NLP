import torch
import torch.nn as nn
import torch.nn.functional as F
import string

# ---------------------------
# 2. Define a decoder-only GPT-like model.
# ---------------------------
class CharGPT(nn.Module):
    def __init__(self, vocab_size, emb_size=128, num_layers=2, num_heads=4, max_seq_length=100):
        """
        GPT-style decoder-only transformer model for character prediction.
        
        Args:
            vocab_size (int): Number of tokens in our vocabulary.
            emb_size (int): Embedding size.
            num_layers (int): Number of transformer decoder layers.
            num_heads (int): Number of attention heads per layer.
            max_seq_length (int): Maximum sequence length for training.
        """
        super(CharGPT, self).__init__()
        self.emb_size = emb_size
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, emb_size))
        
        # Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.ln_f = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, vocab_size)
        self.max_seq_length = max_seq_length

    def forward(self, idx):
        """
        Forward pass for next character prediction.
        
        Args:
            idx (Tensor): Tensor of shape (batch_size, seq_length) containing token indices.
        
        Returns:
            logits (Tensor): Tensor of shape (batch_size, seq_length, vocab_size).
        """
        batch_size, seq_length = idx.size()
        if seq_length > self.max_seq_length:
            raise ValueError("Sequence length exceeds maximum supported length.")

        # Token + Positional embeddings
        token_emb = self.token_embedding(idx)  
        pos_emb = self.pos_embedding[:, :seq_length, :]
        x = token_emb + pos_emb

        # Masked attention to prevent seeing future tokens
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        x = x.transpose(0, 1)  # Transformer expects (seq_length, batch_size, emb_size)
        x = self.decoder(x, x, tgt_mask=causal_mask.to(x.device))
        x = x.transpose(0, 1)  # Back to (batch_size, seq_length, emb_size)
        
        # Final normalization and output layer
        x = self.ln_f(x)
        logits = self.head(x)  # (batch_size, seq_length, vocab_size)
        return logits
    
    @classmethod
    def load_model(cls, model_path, vocab_size, emb_size=128, num_layers=2, num_heads=4, max_seq_length=100, device='cpu'):
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
        model = cls(vocab_size, emb_size, num_layers, num_heads, max_seq_length)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def save(self, model_path):
        """
        Saves the model state to a file.

        Args:
            model_path (str): Path where the model should be saved.
        """
        torch.save(self.state_dict(), model_path)

# ---------------------------
# 3. Prediction function.
# ---------------------------
def get_next_char_predictions(model, context, device='cpu', top_k=3):
    """
    Given the current text context, returns the top-k next character predictions.
    
    Args:
        model (nn.Module): The GPT-like model.
        context (str): The current text sequence.
        device (str): Device on which tensors should reside.
        top_k (int): How many predictions to return.
        
    Returns:
        predictions (list of str): List of the top-k predicted characters.
    """
    indices = [char_to_idx.get(ch, 0) for ch in context]
    if len(indices) == 0:
        indices = [0]
    input_tensor = torch.tensor([indices], dtype=torch.long, device=device)

    # Run model
    logits = model(input_tensor)
    last_logits = logits[0, -1, :]  # Only last step

    # Convert logits to probabilities
    probs = F.softmax(last_logits, dim=0)
    top_probs, top_indices = torch.topk(probs, top_k)
    predictions = [idx_to_char[idx.item()] for idx in top_indices]
    return predictions