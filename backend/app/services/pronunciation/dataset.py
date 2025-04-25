import os
import torch
from torch.utils.data import Dataset
import torchaudio
import re
from collections import Counter

class PronunciationDataset(Dataset):
    def __init__(self, reference_dir, train_dir, max_len=500):
        self.data = []
        self.max_len = max_len
        self.reference_dir = reference_dir
        self.train_dir = train_dir
        
        # Load reference samples (used for comparison)
        self.reference_samples = {}
        for file in os.listdir(reference_dir):
            if file.endswith(".wav"):
                wav_path = os.path.join(reference_dir, file)
                txt_path = wav_path.replace(".wav", ".txt")
                if os.path.exists(txt_path):
                    base_name = os.path.splitext(file)[0]
                    try:
                        # Load the reference audio to verify it's valid
                        waveform, sr = torchaudio.load(wav_path)
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        # Store reference data
                        self.reference_samples[base_name] = {
                            'path': wav_path,
                            'text': text,
                            'waveform': waveform,
                            'sr': sr
                        }
                    except Exception as e:
                        print(f"Skipping invalid reference file {wav_path}: {e}")
                        continue
        
        print(f"Loaded {len(self.reference_samples)} reference samples.")
        
        # Load training samples (correct and incorrect)
        correct_dir = os.path.join(train_dir, "correct")
        incorrect_dir = os.path.join(train_dir, "incorrect")
        
        # Process correct samples (score = 1.0)
        if os.path.exists(correct_dir):
            for file in os.listdir(correct_dir):
                if file.endswith(".wav"):
                    wav_path = os.path.join(correct_dir, file)
                    txt_path = wav_path.replace(".wav", ".txt")
                    if os.path.exists(txt_path):
                        try:
                            # Try to load the audio file to verify it's valid
                            waveform, sr = torchaudio.load(wav_path)
                            # Verify the audio has content
                            if waveform.abs().mean() > 0:
                                self.data.append((wav_path, txt_path, 1.0))
                        except Exception as e:
                            print(f"Skipping invalid correct file {wav_path}: {e}")
                            continue
        
        # Process incorrect samples (score = 0.0)
        if os.path.exists(incorrect_dir):
            for file in os.listdir(incorrect_dir):
                if file.endswith(".wav"):
                    wav_path = os.path.join(incorrect_dir, file)
                    txt_path = wav_path.replace(".wav", ".txt")
                    if os.path.exists(txt_path):
                        try:
                            # Try to load the audio file to verify it's valid
                            waveform, sr = torchaudio.load(wav_path)
                            # Verify the audio has content
                            if waveform.abs().mean() > 0:
                                self.data.append((wav_path, txt_path, 0.0))
                        except Exception as e:
                            print(f"Skipping invalid incorrect file {wav_path}: {e}")
                            continue

        print(f"Loaded {len(self.data)} training samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        wav_path, txt_path, score = self.data[index]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)

        # Convert to MFCC with adjusted parameters to avoid filterbank warnings
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={
                'n_fft': 1024,       # Increased from 400
                'n_mels': 40,        # Reduced from 80
                'hop_length': 512,   # Increased from 160
                'mel_scale': 'htk',
            }
        )
        mfcc = mfcc_transform(waveform).squeeze(0)

        # Pad or truncate
        if mfcc.shape[1] < self.max_len:
            mfcc = torch.nn.functional.pad(mfcc, (0, self.max_len - mfcc.shape[1]))
        else:
            mfcc = mfcc[:, :self.max_len]

        return mfcc, torch.tensor(score, dtype=torch.float32)
    