import os
import torch
from torch.utils.data import Dataset
import torchaudio
import re
from collections import Counter

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
