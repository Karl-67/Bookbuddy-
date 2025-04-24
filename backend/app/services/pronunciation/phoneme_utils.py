import torch
import numpy as np
from typing import Dict, List, Tuple

# Dictionary mapping from English phonemes to indices
# Using the CMU phoneme set (https://cmusphinx.github.io/wiki/userdict/)
CMU_PHONEMES = [
    # Vowels
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    # Consonants
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 
    'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH',
    # Stress markers
    '0', '1', '2',
    # Special tokens
    '<PAD>', '<UNK>'
]

# Create mapping from phoneme to index
PHONEME_TO_IDX = {ph: idx for idx, ph in enumerate(CMU_PHONEMES)}

# Load a simple word-to-phoneme dictionary
def load_cmu_dict(cmu_dict_path: str) -> Dict[str, List[str]]:
    """
    Load the CMU dictionary from a file.
    Returns a dictionary mapping words to their phoneme representations.
    """
    word_to_phonemes = {}
    try:
        with open(cmu_dict_path, 'r', encoding='latin-1') as f:
            for line in f:
                if line.startswith(';;;'):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                    
                word = parts[0].lower()
                # Handle multiple pronunciations by appending (1), (2), etc.
                if '(' in word:
                    word = word.split('(')[0]
                    
                phonemes = parts[1:]
                word_to_phonemes[word] = phonemes
    except FileNotFoundError:
        print(f"Warning: CMU dictionary not found at {cmu_dict_path}.")
        # We'll continue without the dictionary and use a fallback approach
    
    return word_to_phonemes

# Fallback method when CMU dict is not available
def text_to_pseudo_phonemes(text: str) -> List[str]:
    """
    A simple fallback method that creates pseudo-phonemes based on characters.
    This is not linguistically accurate but provides a fallback when a proper
    phoneme dictionary is not available.
    """
    # Remove punctuation and convert to uppercase
    text = ''.join(c for c in text.upper() if c.isalpha() or c.isspace())
    words = text.split()
    
    pseudo_phonemes = []
    for word in words:
        # Create simple pseudo-phonemes by character
        for i in range(len(word)):
            if i < len(word) - 1:
                # Create digraphs for common combinations
                digraph = word[i:i+2]
                if digraph in ['CH', 'SH', 'TH', 'PH', 'WH', 'GH']:
                    pseudo_phonemes.append(digraph)
                    continue
            
            # Add individual character as phoneme
            pseudo_phonemes.append(word[i])
        
        # Add word boundary
        pseudo_phonemes.append('<PAD>')
    
    return pseudo_phonemes

def text_to_phonemes(text: str, cmu_dict: Dict[str, List[str]] = None) -> List[str]:
    """
    Convert text to phonemes using the CMU dictionary.
    Falls back to the pseudo-phoneme approach if the dictionary is not available
    or if a word is not in the dictionary.
    """
    if cmu_dict is None:
        return text_to_pseudo_phonemes(text)
    
    text = text.lower().strip()
    words = text.split()
    phonemes = []
    
    for word in words:
        # Remove punctuation
        clean_word = ''.join(c for c in word if c.isalpha())
        
        if clean_word in cmu_dict:
            phonemes.extend(cmu_dict[clean_word])
        else:
            # Fallback for unknown words
            phonemes.extend(text_to_pseudo_phonemes(clean_word))
        
        # Add word boundary
        phonemes.append('<PAD>')
    
    return phonemes

def phonemes_to_tensor(phonemes: List[str], max_len: int = 100) -> torch.Tensor:
    """
    Convert a list of phonemes to a one-hot tensor.
    """
    # Map phonemes to indices
    indices = [PHONEME_TO_IDX.get(ph, PHONEME_TO_IDX['<UNK>']) for ph in phonemes]
    
    # Truncate or pad
    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices = indices + [PHONEME_TO_IDX['<PAD>']] * (max_len - len(indices))
    
    # Convert to tensor
    phoneme_tensor = torch.zeros(max_len, len(PHONEME_TO_IDX))
    for i, idx in enumerate(indices):
        phoneme_tensor[i, idx] = 1.0
    
    return phoneme_tensor

def get_phoneme_embeddings(vocab_size: int = len(PHONEME_TO_IDX), embedding_dim: int = 64) -> torch.nn.Embedding:
    """
    Create a phoneme embedding layer.
    """
    return torch.nn.Embedding(vocab_size, embedding_dim) 