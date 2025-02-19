import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
from model import CharGPT
from tokenizer import CharTokenizer
from train import train_model
from wiki40B_dataset import clean_text, Wiki40BCharDataset
from tqdm import tqdm
from datasets import load_dataset
#############################
# Preprocessing of the Data #
#############################

device = torch.device("mps")
print(f"Using device: {device}")

# Load the English portion of the wiki40b dataset
dataset = load_dataset("wiki40b", "en")

cleaned_articles = []
# Access the train split and use enumerate
for idx, example in tqdm(enumerate(dataset['train'].select(range(5)))):  
    text = clean_text(example["text"])
    wikidata_id = example["wikidata_id"]
    
    # Skip very short texts
    if len(text) < 100:
        continue
        
    cleaned_articles.append({
        "index": idx, 
        "wikidata_id": wikidata_id, 
        "text": text
    })

# Create dataset
tokenizer = CharTokenizer(languages=['en', 'zh', 'ja', 'fr', 'emoji', 'ar'])

dataset = Wiki40BCharDataset(cleaned_articles[:2], tokenizer, seq_length=25)

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
