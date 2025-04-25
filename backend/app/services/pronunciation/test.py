import os
import sys
import torch
import torchaudio
import numpy as np

# Add the app directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(app_dir)

from services.pronunciation.model import PronunciationClassifier

def load_model(model_path):
    """Load the trained model"""
    model = PronunciationClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model

def process_audio(wav_path, model, max_len=500):
    """Process a single audio file and predict pronunciation score"""
    # Load audio
    waveform, sr = torchaudio.load(wav_path)
    
    # Convert to MFCC with the same parameters used during training
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=13,
        melkwargs={
            'n_fft': 1024,
            'n_mels': 40,
            'hop_length': 512,
            'mel_scale': 'htk',
        }
    )
    mfcc = mfcc_transform(waveform).squeeze(0)
    
    # Pad or truncate
    if mfcc.shape[1] < max_len:
        mfcc = torch.nn.functional.pad(mfcc, (0, max_len - mfcc.shape[1]))
    else:
        mfcc = mfcc[:, :max_len]
    
    # Add batch dimension and transpose for model
    mfcc = mfcc.unsqueeze(0)  # [1, 13, 500]
    mfcc = mfcc.transpose(1, 2)  # [1, 500, 13]
    
    # Get prediction
    with torch.no_grad():
        score = model(mfcc).item()
    
    return score

def main():
    # Path to model checkpoint
    model_path = "models/pronunciation_model_best.pt"
    
    # Path to testing directory on Desktop
    test_dir = os.path.expanduser("~/OneDrive/Desktop/New folder")
    
    # Load model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process all WAV files in the testing directory
    results = []
    
    for file in os.listdir(test_dir):
        if file.endswith(".wav"):
            wav_path = os.path.join(test_dir, file)
            
            # Uncomment text file processing
            txt_path = wav_path.replace(".wav", ".txt")
            transcript = None
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
            
            try:
                # Get pronunciation score
                score = process_audio(wav_path, model)
                results.append((file, score, transcript))
                print(f"File: {file}")
                print(f"Score: {score:.4f} ({score*100:.2f}%)")
                # Uncomment text printing
                if transcript:
                    print(f"Text: {transcript}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Calculate overall statistics
    if results:
        scores = [score for _, score, _ in results]
        print("\nTesting Summary:")
        print(f"Files tested: {len(results)}")
        print(f"Average score: {np.mean(scores):.4f} ({np.mean(scores)*100:.2f}%)")
        print(f"Min score: {np.min(scores):.4f} ({np.min(scores)*100:.2f}%)")
        print(f"Max score: {np.max(scores):.4f} ({np.max(scores)*100:.2f}%)")
        
        # Count files in different score ranges
        excellent = sum(1 for s in scores if s >= 0.9)
        good = sum(1 for s in scores if 0.8 <= s < 0.9)
        average = sum(1 for s in scores if 0.6 <= s < 0.8)
        poor = sum(1 for s in scores if 0.4 <= s < 0.6)
        very_poor = sum(1 for s in scores if s < 0.4)
        
        print("\nScore Distribution:")
        print(f"Excellent (90-100%): {excellent} files")
        print(f"Good (80-90%): {good} files")
        print(f"Average (60-80%): {average} files")
        print(f"Poor (40-60%): {poor} files")
        print(f"Very Poor (0-40%): {very_poor} files")

if __name__ == "__main__":
    main() 