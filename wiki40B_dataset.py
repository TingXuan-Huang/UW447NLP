from datasets import load_dataset
import torch
import re
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader

class Wiki40BCharDataset(Dataset):
    def __init__(self, cleaned_articles, tokenizer, seq_length=100):
        """
        A character-level dataset for Wiki40B articles.

        Args:
            cleaned_articles (list): List of dictionaries containing cleaned articles
                                   with 'text', 'index', and 'wikidata_id' keys
            tokenizer (CharTokenizer): The tokenizer to use for encoding
            seq_length (int): The length of input sequences
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Process each article separately and store sequences
        self.sequences = []
        
        for article in cleaned_articles:
            text = article['text']
            
            # Skip articles that are too short
            if len(text) < seq_length:
                continue
                
            # Create sequences for this article
            for i in range(0, len(text) - seq_length):
                input_text = text[i: i + seq_length]
                target_text = text[i + 1: i + seq_length + 1]
                
                # Encode the input and target texts
                input_indices = torch.tensor(self.tokenizer.encode(input_text), dtype=torch.long)
                target_indices = torch.tensor(self.tokenizer.encode(target_text), dtype=torch.long)
                
                # Add special tokens
                input_indices = torch.cat((
                    torch.tensor([self.tokenizer.special_tokens["<BOS>"]], dtype=torch.long),
                    input_indices
                ))
                target_indices = torch.cat((
                    target_indices,
                    torch.tensor([self.tokenizer.special_tokens["<EOS>"]], dtype=torch.long)
                ))
                
                # Truncate if necessary
                if len(input_indices) > self.seq_length:
                    input_indices = input_indices[:self.seq_length]
                if len(target_indices) > self.seq_length:
                    target_indices = target_indices[:self.seq_length]
                
                # Store the sequence pair
                self.sequences.append((input_indices, target_indices))
        
        # print(f"Created {len(self.sequences)} sequences from {len(cleaned_articles)} articles")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns a pre-computed (input sequence, target sequence) pair.
        """
        return self.sequences[idx]

def clean_text(text):
    """
    Cleans individual Wiki-40B text by removing special markers and formatting
    """
    # Remove special markers
    text = re.sub(r"_START_ARTICLE_|_START_SECTION_|_START_PARAGRAPH_", "", text)
    
    # Replace `_NEWLINE_` with actual newlines
    text = text.replace("_NEWLINE_", "\n")
    
    # Remove multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    
    return text