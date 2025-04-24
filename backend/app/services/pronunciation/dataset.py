import os
import torch
from torch.utils.data import Dataset
import torchaudio
import re
from collections import Counter
from typing import Tuple, Optional, List
import numpy as np
import torch.nn as nn

# Import phoneme utilities if available
try:
    from .phoneme_utils import text_to_phonemes, load_cmu_dict, PHONEME_TO_IDX
    PHONEME_UTILS_AVAILABLE = True
except ImportError:
    PHONEME_UTILS_AVAILABLE = False
    print("Warning: Phoneme utilities not available. Some functionality will be limited.")

class PronunciationDataset(Dataset):
    def __init__(self, root_dir, max_len=500):
        self.data = []
        self.max_len = max_len
        self.root_dir = root_dir
        skipped_files = 0
        
        # Common difficult phonemes in English
        self.difficult_phonemes = {
            'th', 'ch', 'sh', 'zh', 'ng', 'r', 'l', 'w', 'y',
            'ae', 'ai', 'ao', 'au', 'ei', 'oi', 'ou', 'uh'
        }
        
        # Load all text files first to calculate statistics
        all_texts = []
        for speaker in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker)
            if not os.path.isdir(speaker_path) or speaker.startswith('.'):
                continue

            for file in os.listdir(speaker_path):
                if file.endswith(".wav") and not file.startswith("._"):
                    wav_path = os.path.join(speaker_path, file)
                    txt_path = wav_path.replace(".wav", ".txt")
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                all_texts.append(text)
                        except Exception as e:
                            skipped_files += 1
                            continue

        # Calculate word frequency statistics
        all_words = ' '.join(all_texts).lower().split()
        word_freq = Counter(all_words)
        max_freq = max(word_freq.values())
        
        # Now process the actual dataset
        for speaker in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker)
            if not os.path.isdir(speaker_path) or speaker.startswith('.'):
                continue

            for file in os.listdir(speaker_path):
                if file.endswith(".wav") and not file.startswith("._"):
                    wav_path = os.path.join(speaker_path, file)
                    txt_path = wav_path.replace(".wav", ".txt")
                    if os.path.exists(txt_path):
                        try:
                            # Try to load the audio file to verify it's valid
                            waveform, sr = torchaudio.load(wav_path)
                            # Verify the audio has content
                            if waveform.abs().mean() > 0:
                                self.data.append((wav_path, txt_path))
                            else:
                                skipped_files += 1
                        except Exception as e:
                            skipped_files += 1
                            continue

        print(f"Loaded {len(self.data)} valid samples. Skipped {skipped_files} invalid files.")
        self.word_freq = word_freq
        self.max_freq = max_freq

    def __len__(self):
        return len(self.data)

    def calculate_pronunciation_score(self, text):
        words = text.lower().split()
        if len(words) == 0:
            return 0.5
            
        # Calculate various factors that affect pronunciation difficulty
        scores = []
        
        for word in words:
            word_score = 0.0
            factors = 0
            
            # Factor 1: Word length (longer words tend to be harder)
            word_score += min(1.0, len(word) / 12)  # Normalize by max expected length
            factors += 1
            
            # Factor 2: Word frequency (rarer words are harder)
            freq = self.word_freq.get(word, 1)
            word_score += 1.0 - (freq / self.max_freq)  # Rarer words get higher scores
            factors += 1
            
            # Factor 3: Presence of difficult phonemes
            difficult_phoneme_count = sum(1 for p in self.difficult_phonemes if p in word)
            word_score += min(1.0, difficult_phoneme_count / 3)  # Normalize by max expected difficult phonemes
            factors += 1
            
            # Average the factors
            scores.append(word_score / factors)
        
        # Calculate final score (0.0 to 1.0)
        final_score = sum(scores) / len(scores)
        
        # Scale to our desired range (0.5 to 1.0)
        return 0.5 + (final_score * 0.5)

    def __getitem__(self, index):
        wav_path, txt_path = self.data[index]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)

        # Convert to MFCC with adjusted parameters
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={
                'n_fft': 400,
                'n_mels': 80,
                'hop_length': 160,
                'mel_scale': 'htk',
            }
        )
        mfcc = mfcc_transform(waveform).squeeze(0)

        # Pad or truncate
        if mfcc.shape[1] < self.max_len:
            mfcc = torch.nn.functional.pad(mfcc, (0, self.max_len - mfcc.shape[1]))
        else:
            mfcc = mfcc[:, :self.max_len]

        # Read the text file
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Calculate pronunciation score using our improved method
        score = self.calculate_pronunciation_score(text)

        return mfcc, torch.tensor(score, dtype=torch.float32)


class Wav2Vec2PronunciationDataset(Dataset):
    def __init__(self, root_dir, target_sr=16000, max_len=160000):
        self.data = []
        self.max_len = max_len  # Max length in samples (10 seconds at 16kHz)
        self.target_sr = target_sr
        self.root_dir = root_dir
        
        # Process dataset
        skipped_files = 0
        for speaker in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker)
            if not os.path.isdir(speaker_path) or speaker.startswith('.'):
                continue

            for file in os.listdir(speaker_path):
                if file.endswith(".wav") and not file.startswith("._"):
                    wav_path = os.path.join(speaker_path, file)
                    txt_path = wav_path.replace(".wav", ".txt")
                    if os.path.exists(txt_path):
                        try:
                            # Try to load the audio file to verify it's valid
                            waveform, sr = torchaudio.load(wav_path)
                            # Verify the audio has content
                            if waveform.abs().mean() > 0:
                                self.data.append((wav_path, txt_path))
                            else:
                                skipped_files += 1
                        except Exception as e:
                            skipped_files += 1
                            continue

        print(f"Loaded {len(self.data)} valid samples. Skipped {skipped_files} invalid files.")
        
        # Initialize phoneme map for future extension (phoneme embedding approach)
        self.phoneme_map = {}

    def __len__(self):
        return len(self.data)

    def calculate_pronunciation_score(self, text):
        # Simple proxy for pronunciation difficulty
        word_count = len(text.split())
        unique_chars = len(set(text.lower()))
        
        # Normalize to [0, 1] range - more words and unique chars = harder pronunciation
        word_factor = min(1.0, word_count / 15)  # Cap at 15 words
        char_factor = min(1.0, unique_chars / 30)  # Cap at 30 unique chars
        
        # Simple scoring formula
        score = (word_factor + char_factor) / 2
        
        # Scale to [0.5, 1.0] range to match original dataset
        return 0.5 + (score * 0.5)

    def __getitem__(self, index):
        wav_path, txt_path = self.data[index]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        
        # Normalize waveform
        waveform = waveform / (waveform.abs().max() + 1e-7)
        
        # Pad or truncate
        waveform = waveform.squeeze(0)  # Remove channel dimension
        if waveform.shape[0] < self.max_len:
            # Pad with silence
            padding = self.max_len - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Randomly crop
            start = torch.randint(0, waveform.shape[0] - self.max_len + 1, (1,)).item()
            waveform = waveform[start:start + self.max_len]

        # Read the text file
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Calculate pronunciation score
        score = self.calculate_pronunciation_score(text)

        return waveform, torch.tensor(score, dtype=torch.float32)


class PhonemeDataset(Dataset):
    def __init__(self, root_dir, cmu_dict_path=None, max_audio_len=500, max_phoneme_len=100):
        self.data = []
        self.max_audio_len = max_audio_len
        self.max_phoneme_len = max_phoneme_len
        self.root_dir = root_dir
        
        # Try to load CMU dictionary if available
        self.cmu_dict = None
        if PHONEME_UTILS_AVAILABLE and cmu_dict_path and os.path.exists(cmu_dict_path):
            print(f"Loading CMU dictionary from {cmu_dict_path}...")
            self.cmu_dict = load_cmu_dict(cmu_dict_path)
            print(f"Loaded {len(self.cmu_dict)} entries from CMU dictionary")
        else:
            print("CMU dictionary not available. Using fallback method for phoneme conversion.")
        
        # Process dataset
        skipped_files = 0
        for speaker in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker)
            if not os.path.isdir(speaker_path) or speaker.startswith('.'):
                continue

            for file in os.listdir(speaker_path):
                if file.endswith(".wav") and not file.startswith("._"):
                    wav_path = os.path.join(speaker_path, file)
                    txt_path = wav_path.replace(".wav", ".txt")
                    if os.path.exists(txt_path):
                        try:
                            # Try to load the audio file to verify it's valid
                            waveform, sr = torchaudio.load(wav_path)
                            # Verify the audio has content
                            if waveform.abs().mean() > 0:
                                self.data.append((wav_path, txt_path))
                            else:
                                skipped_files += 1
                        except Exception as e:
                            skipped_files += 1
                            continue

        print(f"Loaded {len(self.data)} valid samples. Skipped {skipped_files} invalid files.")

    def __len__(self):
        return len(self.data)
    
    def text_to_phoneme_indices(self, text: str) -> torch.Tensor:
        """Convert text to phoneme indices"""
        if not PHONEME_UTILS_AVAILABLE:
            # Fallback when phoneme utils are not available
            chars = [c for c in text.upper() if c.isalpha() or c.isspace()]
            indices = [ord(c) % 50 for c in chars]  # Simple hash to stay within phoneme vocab size
            
            # Pad or truncate
            if len(indices) > self.max_phoneme_len:
                indices = indices[:self.max_phoneme_len]
            else:
                indices = indices + [0] * (self.max_phoneme_len - len(indices))
                
            return torch.tensor(indices, dtype=torch.long)
            
        # Convert text to phonemes using utility
        phonemes = text_to_phonemes(text, self.cmu_dict)
        
        # Convert phonemes to indices
        indices = [PHONEME_TO_IDX.get(ph, PHONEME_TO_IDX['<UNK>']) for ph in phonemes]
        
        # Pad or truncate
        if len(indices) > self.max_phoneme_len:
            indices = indices[:self.max_phoneme_len]
        else:
            indices = indices + [PHONEME_TO_IDX['<PAD>']] * (self.max_phoneme_len - len(indices))
            
        return torch.tensor(indices, dtype=torch.long)

    def calculate_pronunciation_score(self, text: str) -> float:
        """Calculate pronunciation score based on text complexity"""
        words = text.lower().split()
        if len(words) == 0:
            return 0.5
            
        # Calculate difficulty score
        avg_word_len = sum(len(w) for w in words) / len(words)
        unique_chars = len(set(text.lower()))
        
        # Length and complexity factors
        length_factor = min(1.0, avg_word_len / 10)
        complexity_factor = min(1.0, unique_chars / 30)
        
        # Calculate final score (0.0 to 1.0)
        score = (length_factor + complexity_factor) / 2
        
        # Scale to our desired range (0.5 to 1.0)
        return 0.5 + (score * 0.5)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wav_path, txt_path = self.data[index]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)

        # Convert to MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={
                'n_fft': 400,
                'n_mels': 80,
                'hop_length': 160,
                'mel_scale': 'htk',
            }
        )
        mfcc = mfcc_transform(waveform).squeeze(0)

        # Pad or truncate MFCC
        if mfcc.shape[1] < self.max_audio_len:
            mfcc = torch.nn.functional.pad(mfcc, (0, self.max_audio_len - mfcc.shape[1]))
        else:
            mfcc = mfcc[:, :self.max_audio_len]

        # Read the text file
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Convert text to phoneme indices
        phoneme_indices = self.text_to_phoneme_indices(text)
        
        # Calculate pronunciation score
        score = self.calculate_pronunciation_score(text)

        return mfcc, phoneme_indices, torch.tensor(score, dtype=torch.float32)


class ReferencePronunciationDataset(Dataset):
    """
    Dataset for pronunciation evaluation that uses reference (native speaker) audio
    as ground truth for comparison.
    """
    def __init__(self, user_audio_dir, reference_audio_dir, max_len=500):
        """
        Initialize dataset with both user and reference audio directories.
        
        Args:
            user_audio_dir: Directory containing user audio recordings
            reference_audio_dir: Directory containing reference audio recordings
            max_len: Maximum length of MFCC features
        """
        self.data = []
        self.max_len = max_len
        self.user_audio_dir = user_audio_dir
        self.reference_audio_dir = reference_audio_dir
        skipped_files = 0
        
        # Map of text to reference audio paths
        self.reference_map = {}
        
        # First, build reference map
        for file in os.listdir(reference_audio_dir):
            if file.endswith(".wav") and not file.startswith("._"):
                ref_wav_path = os.path.join(reference_audio_dir, file)
                ref_txt_path = ref_wav_path.replace(".wav", ".txt")
                
                if os.path.exists(ref_txt_path):
                    try:
                        with open(ref_txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip().lower()
                            # Index by text content for lookup
                            self.reference_map[text] = ref_wav_path
                    except Exception as e:
                        print(f"Error reading reference text file {ref_txt_path}: {e}")
                        continue
        
        print(f"Loaded {len(self.reference_map)} reference pronunciations")
        
        # Now process user audio files
        for speaker in os.listdir(user_audio_dir):
            speaker_path = os.path.join(user_audio_dir, speaker)
            if not os.path.isdir(speaker_path) or speaker.startswith('.'):
                continue

            for file in os.listdir(speaker_path):
                if file.endswith(".wav") and not file.startswith("._"):
                    wav_path = os.path.join(speaker_path, file)
                    txt_path = wav_path.replace(".wav", ".txt")
                    
                    if os.path.exists(txt_path):
                        try:
                            # Read text content
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip().lower()
                            
                            # Check if we have a reference for this text
                            ref_path = self.reference_map.get(text)
                            if ref_path is None:
                                # Try finding a reference that contains this text
                                for ref_text, rp in self.reference_map.items():
                                    if text in ref_text:
                                        ref_path = rp
                                        break
                            
                            # If we have a reference, add this sample to the dataset
                            if ref_path is not None:
                                # Try to load the audio file to verify it's valid
                                waveform, sr = torchaudio.load(wav_path)
                                # Verify the audio has content
                                if waveform.abs().mean() > 0:
                                    self.data.append((wav_path, txt_path, ref_path))
                                else:
                                    skipped_files += 1
                            else:
                                # No reference found, skip this file
                                skipped_files += 1
                                
                        except Exception as e:
                            skipped_files += 1
                            print(f"Error processing user audio file {wav_path}: {e}")
                            continue

        print(f"Loaded {len(self.data)} valid user samples with references. Skipped {skipped_files} invalid/unmatched files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_wav_path, txt_path, ref_wav_path = self.data[index]

        # Load user audio and convert to MFCC
        user_waveform, user_sr = torchaudio.load(user_wav_path)
        user_mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=user_sr,
            n_mfcc=13,
            melkwargs={
                'n_fft': 400,
                'n_mels': 80,
                'hop_length': 160,
                'mel_scale': 'htk',
            }
        )
        user_mfcc = user_mfcc_transform(user_waveform).squeeze(0)

        # Load reference audio and convert to MFCC
        ref_waveform, ref_sr = torchaudio.load(ref_wav_path)
        ref_mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=ref_sr,
            n_mfcc=13,
            melkwargs={
                'n_fft': 400,
                'n_mels': 80,
                'hop_length': 160,
                'mel_scale': 'htk',
            }
        )
        ref_mfcc = ref_mfcc_transform(ref_waveform).squeeze(0)

        # Pad or truncate user MFCC
        if user_mfcc.shape[1] < self.max_len:
            user_mfcc = torch.nn.functional.pad(user_mfcc, (0, self.max_len - user_mfcc.shape[1]))
        else:
            user_mfcc = user_mfcc[:, :self.max_len]

        # Pad or truncate reference MFCC
        if ref_mfcc.shape[1] < self.max_len:
            ref_mfcc = torch.nn.functional.pad(ref_mfcc, (0, self.max_len - ref_mfcc.shape[1]))
        else:
            ref_mfcc = ref_mfcc[:, :self.max_len]

        # Calculate similarity score between user and reference
        similarity_score = self.calculate_audio_similarity(user_mfcc, ref_mfcc)

        return user_mfcc, ref_mfcc, torch.tensor(similarity_score, dtype=torch.float32)

    def calculate_audio_similarity(self, user_mfcc, ref_mfcc):
        """
        Calculate similarity between user and reference MFCCs.
        Returns a score between 0 and 1, where 1 is perfect match.
        """
        # Calculate mean squared error between MFCCs
        mse = torch.mean((user_mfcc - ref_mfcc) ** 2).item()
        
        # Transform MSE to a similarity score (0 to 1)
        # Lower MSE = higher similarity
        similarity = 1.0 / (1.0 + mse)
        
        # Scale to [0.2, 1.0] range to avoid extreme penalties
        scaled_similarity = 0.2 + (similarity * 0.8)
        
        return scaled_similarity

# Add a class for inference with reference audio
class ReferenceBasedPronunciationModel(nn.Module):
    """
    Model that directly compares user audio with reference audio
    to evaluate pronunciation quality.
    """
    def __init__(self, input_dim=13, hidden_dim=128):
        super().__init__()
        
        # Siamese network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Comparison network
        self.comparison = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, user_audio, reference_audio):
        # Extract features from both inputs
        user_features = self.feature_extractor(user_audio)
        ref_features = self.feature_extractor(reference_audio)
        
        # Global average pooling over time dimension
        user_features = torch.mean(user_features, dim=1)
        ref_features = torch.mean(ref_features, dim=1)
        
        # Concatenate features
        combined = torch.cat([user_features, ref_features], dim=1)
        
        # Calculate similarity score
        similarity = self.comparison(combined)
        
        return similarity
