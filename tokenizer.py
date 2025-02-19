from transformers import CanineTokenizer
import string
import unicodedata
import torch

class CharTokenizer:
    def __init__(self, languages=['en']):
        """
        Initialize with a filtered character set using Unicode ranges.
        
        Args:
            languages (list): List of language codes to include:
                'en' - English (Latin)
                'zh' - Chinese (Mandarin)
                'hi' - Hindi (Devanagari)
                'es' - Spanish (Latin + extras)
                'ar' - Arabic
                'bn' - Bengali
                'pt' - Portuguese (Latin + extras)
                'ur' - Urdu
                'id' - Indonesian (Latin)
                'fr' - French (Latin + extras)
                'de' - German (Latin + extras)
                'ja' - Japanese
                'emoji' - Emojis
        """
        # Initialize special tokens
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<NEWLINE>": 4,  # Define a special token for newlines
        }
        
        # Initialize character sets
        self.char_to_id = self.special_tokens.copy()
        self.next_id = len(self.special_tokens)
        
        # Unicode ranges for different languages
        unicode_ranges = {
            'en': [
                (0x0020, 0x007F),  # Basic Latin (ASCII)
            ],
            'es': [
                (0x0020, 0x007F),  # Basic Latin
                (0x00C0, 0x00FF),  # Latin-1 Supplement (áéíóúñ etc.)
            ],
            'pt': [
                (0x0020, 0x007F),  # Basic Latin
                (0x00C0, 0x00FF),  # Latin-1 Supplement (ãõáéí etc.)
            ],
            'fr': [
                (0x0020, 0x007F),  # Basic Latin
                (0x00C0, 0x00FF),  # Latin-1 Supplement (éèêë etc.)
            ],
            'de': [
                (0x0020, 0x007F),  # Basic Latin
                (0x00C0, 0x00FF),  # Latin-1 Supplement (äöüß etc.)
            ],
            'zh': [
                (0x4E00, 0x9FFF),  # CJK Unified Ideographs
                (0x3000, 0x303F),  # CJK Symbols and Punctuation
                (0xFF00, 0xFFEF),  # Fullwidth forms
                (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
                (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
                (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
                (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
                (0x2F00, 0x2FDF),   # Kangxi Radicals
                (0x2E80, 0x2EFF),   # CJK Radicals Supplement
                (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
                (0x2000, 0x206F),   # General Punctuation
                (0xFE30, 0xFE4F),   # CJK Compatibility Forms
            ],
            'hi': [
                (0x0900, 0x097F),  # Devanagari
                (0x0020, 0x007F),  # Basic Latin
            ],
            'ar': [
                (0x0600, 0x06FF),  # Arabic
                (0x0750, 0x077F),  # Arabic Supplement
                (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
                (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
            ],
            'bn': [
                (0x0980, 0x09FF),  # Bengali
                (0x0020, 0x007F),  # Basic Latin
            ],
            'ur': [
                (0x0600, 0x06FF),  # Arabic (Urdu uses Arabic script)
                (0x0750, 0x077F),  # Arabic Supplement
                (0x0020, 0x007F),  # Basic Latin
            ],
            'id': [
                (0x0020, 0x007F),  # Basic Latin
                (0x00C0, 0x00FF),  # Latin-1 Supplement
            ],
            'ja': [
                (0x3040, 0x309F),  # Hiragana
                (0x30A0, 0x30FF),  # Katakana
                (0x4E00, 0x9FFF),  # CJK Unified Ideographs
                (0x3000, 0x303F),  # CJK Symbols and Punctuation
                (0xFF00, 0xFFEF),  # Fullwidth forms
                (0x2000, 0x206F),  # General Punctuation
                (0xFE30, 0xFE4F),  # CJK Compatibility Forms
            ],
            'emoji': [
                (0x1F300, 0x1F9FF),  # Emoji & Pictographs
                (0x1F600, 0x1F64F),  # Emoticons
                (0x2600, 0x26FF),    # Miscellaneous Symbols
                (0x2700, 0x27BF),    # Dingbats
            ]
        }
        
        # Add characters for selected languages
        for lang in languages:
            if lang in unicode_ranges:
                for start, end in unicode_ranges[lang]:
                    for code_point in range(start, end + 1):
                        try:
                            char = chr(code_point)
                            # Skip control characters and non-printable characters
                            if not unicodedata.category(char).startswith('C'):
                                if char not in self.char_to_id:
                                    self.char_to_id[char] = self.next_id
                                    self.next_id += 1
                        except ValueError:
                            continue
        
        # Create reverse mapping
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
    def encode(self, text):
        """Convert text to token ids."""
        
        ids = []
            
        for char in text:
            if char == '\n':
                token_id = self.special_tokens["<NEWLINE>"]  # Use the newline token
            else:
                token_id = self.char_to_id.get(char, self.special_tokens["<UNK>"])
            ids.append(token_id)
        
        return ids
    
    def decode(self, ids):
        """Convert token ids back to text."""
        text = ""
        
        # Check if the input is a tensor and flatten it if necessary
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()  # Flatten the tensor and convert to list

        for id in ids:
            if id in [self.special_tokens["<PAD>"], 
                       self.special_tokens["<BOS>"], 
                       self.special_tokens["<EOS>"]]:
                continue
            
            if id == self.special_tokens["<NEWLINE>"]:
                text += '\n'  # Replace the newline token with an actual newline
            else:
                char = self.id_to_char.get(id, "<UNK>")
                text += char
            
        return text
    
    @property
    def vocab_size(self):
        return len(self.char_to_id)
    
    def display_vocab(self):
        """Display the current vocabulary grouped by Unicode blocks."""
        print(f"\nVocabulary size: {self.vocab_size}")
        
        # Group characters by Unicode block
        blocks = {}
        for char, id in sorted(self.char_to_id.items()):
            if char in self.special_tokens.keys():
                continue
            try:
                block = unicodedata.name(char).split()[0]
                if block not in blocks:
                    blocks[block] = []
                blocks[block].append((char, id))
            except ValueError:
                continue
        
        # Print special tokens first
        print("\nSpecial tokens:")
        for token, id in self.special_tokens.items():
            print(f"{token}: {id}")
        
        # Print characters by block
        print("\nCharacter mappings by Unicode block:")
        for block, chars in sorted(blocks.items()):
            print(f"\n{block}:")
            for char, id in chars:
                print(f"  '{char}' ({unicodedata.name(char, 'UNKNOWN')}) -> {id}")

# Example usage
if __name__ == "__main__":
    # Test with all languages
    tokenizer = CharTokenizer(
        languages=['en', 'zh', 'hi', 'es', 'ar', 'bn', 'pt', 'ur', 'id', 'fr', 'de', 'ja', 'emoji']
    )
    
    # Test texts
    texts = [
        "Hello, World! ",
        "你好，世界！",           # Chinese
        "こんにちは世界！",       # Japanese
        """复杂的词汇导致复杂的思维。
        難しい言葉は難しい考えにつながる。
        """
    ]
    
    # Test each text
    for text in texts:
        print("\n" + "="*50)
        print(f"Testing text: {text}")
        print("\nOriginal text character codes:")
        for char in text:
            print(f"'{char}' (U+{ord(char):04X})")
            
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"\nEncoded: {encoded}")
        print(f"Decoded: {decoded}")
        print(f"Roundtrip successful: {text == decoded}")
        if text != decoded:
            print("\nDifferences found:")
            for i, (c1, c2) in enumerate(zip(text, decoded)):
                if c1 != c2:
                    print(f"Position {i}: '{c1}' (U+{ord(c1):04X}) != '{c2}' (U+{ord(c2):04X})")
    