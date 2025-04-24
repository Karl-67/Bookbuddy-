import torch
import os
import numpy as np
from typing import List, Dict, Tuple
from .audio_utils import load_audio
from .feature_extractor import extract_mfcc
from .model import (
    PronunciationClassifier, 
    TransferLearningPronunciationClassifier,
    PhonemeEmbeddingPronunciationClassifier,
    RegressionPronunciationClassifier,
    ReferenceBasedPronunciationModel
)
import torchaudio
import librosa

# Try to import phoneme utils
try:
    from .phoneme_utils import text_to_phonemes, load_cmu_dict, PHONEME_TO_IDX
    PHONEME_UTILS_AVAILABLE = True
except ImportError:
    PHONEME_UTILS_AVAILABLE = False


def get_available_models():
    """Get a list of all available pronunciation models in the models directory."""
    model_path = "models"
    available_models = []
    
    if not os.path.exists(model_path):
        return available_models
    
    for file in os.listdir(model_path):
        if file.startswith("pronunciation_") and file.endswith(".pt"):
            # Extract model type from filename
            parts = file.split("_")
            if len(parts) >= 2:
                model_type = parts[1]
                if model_type not in [m["type"] for m in available_models]:
                    available_models.append({
                        "type": model_type,
                        "path": os.path.join(model_path, file)
                    })
    
    return available_models


def segment_audio_by_silence(audio: np.ndarray, sr: int = 16000, 
                             min_silence_duration: float = 0.3, 
                             silence_threshold: float = 0.03) -> List[Tuple[int, int]]:
    """
    Segment audio based on silence detection.
    Returns a list of (start_sample, end_sample) tuples.
    """
    # Convert silence duration to samples
    min_silence_samples = int(min_silence_duration * sr)
    
    # Calculate audio energy
    energy = librosa.feature.rms(y=audio)[0]
    
    # Normalize energy to 0-1 range
    if energy.max() > 0:
        energy = energy / energy.max()
        
    # Find silent regions (energy below threshold)
    silent = energy < silence_threshold
    
    # Find boundaries of silent regions
    silent_regions = []
    in_silence = False
    silence_start = 0
    
    # Apply a frame-to-sample ratio to convert frame indices to sample indices
    frame_length = len(audio) // len(energy)
    
    for i, is_silent in enumerate(silent):
        if is_silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = i
        elif not is_silent and in_silence:
            # End of silence
            silence_end = i
            if (silence_end - silence_start) * frame_length >= min_silence_samples:
                silent_regions.append((silence_start * frame_length, silence_end * frame_length))
            in_silence = False
    
    # Use silent regions to find non-silent segments (words)
    segments = []
    prev_end = 0
    
    # Add initial segment if audio doesn't start with silence
    if len(silent_regions) == 0 or silent_regions[0][0] > 0:
        start = 0
        end = len(audio) if len(silent_regions) == 0 else silent_regions[0][0]
        segments.append((start, end))
        prev_end = end
    
    # Process segments between silent regions
    for i in range(len(silent_regions)):
        start = silent_regions[i][1]
        if i < len(silent_regions) - 1:
            end = silent_regions[i+1][0]
        else:
            end = len(audio)
        
        # Only add if it's a substantial segment
        if end - start > min_silence_samples:
            segments.append((start, end))
            prev_end = end
    
    # Add final segment if needed
    if prev_end < len(audio) - min_silence_samples:
        segments.append((prev_end, len(audio)))
    
    return segments


def align_segments_with_words(segments: List[Tuple[int, int]], text: str) -> List[Tuple[Tuple[int, int], str]]:
    """
    Align audio segments with words in text.
    Simple approach: just match segments to words in order.
    
    Returns a list of ((start_sample, end_sample), word) tuples.
    """
    words = text.strip().split()
    aligned = []
    
    # If we have more segments than words, merge some segments
    if len(segments) > len(words):
        # Merge segments to match word count
        merged_segments = []
        segments_per_word = len(segments) // len(words)
        remainder = len(segments) % len(words)
        
        segment_idx = 0
        for i in range(len(words)):
            # Calculate how many segments to merge for this word
            count = segments_per_word + (1 if i < remainder else 0)
            start = segments[segment_idx][0]
            end = segments[segment_idx + count - 1][1]
            merged_segments.append((start, end))
            segment_idx += count
        
        segments = merged_segments
    
    # If we have more words than segments, merge some words
    elif len(words) > len(segments):
        # Merge words to match segment count
        merged_words = []
        words_per_segment = len(words) // len(segments)
        remainder = len(words) % len(segments)
        
        word_idx = 0
        for i in range(len(segments)):
            # Calculate how many words to merge for this segment
            count = words_per_segment + (1 if i < remainder else 0)
            merged_word = " ".join(words[word_idx:word_idx + count])
            merged_words.append(merged_word)
            word_idx += count
        
        words = merged_words
    
    # Now we have equal numbers of segments and words, so align them
    for i in range(min(len(segments), len(words))):
        aligned.append((segments[i], words[i]))
    
    return aligned


def check_pronunciation_with_reference(audio_path: str, text: str, reference_path: str = None) -> dict:
    """
    Evaluate pronunciation by comparing with a reference audio.
    
    Args:
        audio_path: Path to user audio file
        text: Transcription text
        reference_path: Path to reference audio (if None, will try to find a matching reference)
    """
    # Load available references
    reference_dir = os.environ.get("REFERENCE_AUDIO_DIR", "data/references")
    
    # If no specific reference is provided, try to find one based on text
    if reference_path is None:
        if not os.path.exists(reference_dir):
            return {"error": f"Reference directory {reference_dir} not found"}
            
        # Look for a reference with matching text
        for file in os.listdir(reference_dir):
            if file.endswith(".txt"):
                txt_path = os.path.join(reference_dir, file)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip().lower()
                
                # Check if texts match or if reference contains the user text
                if text.lower() == ref_text or text.lower() in ref_text:
                    reference_path = os.path.join(reference_dir, file.replace(".txt", ".wav"))
                    if os.path.exists(reference_path):
                        break
        
        if reference_path is None:
            return {"error": "No matching reference found for the provided text"}
    
    # Load user audio
    user_audio = load_audio(audio_path)
    
    # Load reference audio
    try:
        ref_audio, _ = torchaudio.load(reference_path)
        ref_audio = ref_audio.numpy().squeeze()
    except Exception as e:
        return {"error": f"Failed to load reference audio: {str(e)}"}
        
    # Compare user and reference audio using DTW
    similarity_score = compute_audio_similarity(user_audio, ref_audio)
    
    # Try to load reference-based model if available
    model_path = "models/pronunciation_reference_best.pt"
    model_score = None
    
    if os.path.exists(model_path):
        try:
            # Load the model
            model = ReferenceBasedPronunciationModel()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
            model.eval()
            
            # Process audio to MFCC
            user_mfcc = extract_mfcc(user_audio)
            ref_mfcc = extract_mfcc(ref_audio)
            
            # Convert to tensors
            user_tensor = torch.tensor(user_mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            ref_tensor = torch.tensor(ref_mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            
            # Get model prediction
            with torch.no_grad():
                model_score = model(user_tensor, ref_tensor).item()
        except Exception as e:
            print(f"Error using reference model: {e}")
    
    # Combine both scores if available, otherwise use similarity score
    final_score = model_score if model_score is not None else similarity_score
    
    # Return simplified binary classification output
    return {
        "status": "Correct" if final_score > 0.5 else "Incorrect",
        "model_type": "reference"
    }


def compute_audio_similarity(user_audio, reference_audio):
    """
    Compute similarity between user and reference audio using DTW on MFCCs.
    Returns a score between 0 and 1, where 1 is perfect match.
    """
    # Convert audio to MFCC features
    user_mfcc = extract_mfcc(user_audio)
    ref_mfcc = extract_mfcc(reference_audio)
    
    # Compute distance using Dynamic Time Warping
    try:
        # Compute DTW distance
        distance, _ = librosa.sequence.dtw(user_mfcc, ref_mfcc, metric='euclidean')
        
        # Normalize distance to a similarity score (higher is better)
        # Scale distance exponentially to a range between 0 and 1
        # Small distances give scores close to 1, large distances close to 0
        normalized_distance = distance[-1, -1] / (len(user_mfcc) + len(ref_mfcc))
        similarity = np.exp(-normalized_distance)
        
        # Scale to avoid extremely low scores
        similarity = max(0.2, similarity)
        similarity = min(1.0, similarity)
        
        return similarity
    except Exception as e:
        print(f"Error computing DTW: {e}")
        # Fall back to simpler metric
        mse = np.mean((user_mfcc[:min(len(user_mfcc), len(ref_mfcc))] - 
                       ref_mfcc[:min(len(user_mfcc), len(ref_mfcc))]) ** 2)
        similarity = 1.0 / (1.0 + mse)
        return min(1.0, max(0.2, similarity * 0.8 + 0.2))


def check_pronunciation_word_by_word(audio_path: str, text: str, model_type: str = None, reference_path: str = None) -> dict:
    """
    Evaluate pronunciation for each word in the input text.
    Returns binary classification for individual words and the overall utterance.
    
    Args:
        audio_path: Path to audio file
        text: Transcription text 
        model_type: Optional model type to use (wav2vec2, mfcc, phoneme, regression, reference)
        reference_path: Optional path to reference audio (for reference-based evaluation)
    """
    # If using reference-based evaluation
    if model_type == "reference" or (model_type is None and os.path.exists("models/pronunciation_reference_best.pt")):
        if reference_path is not None:
            # Use reference-based evaluation
            overall_score = check_pronunciation_with_reference(audio_path, text, reference_path)
            
            # Load the audio files
            user_audio = load_audio(audio_path)
            ref_audio, _ = torchaudio.load(reference_path)
            ref_audio = ref_audio.numpy().squeeze()
            
            # Segment both audio files
            user_segments = segment_audio_by_silence(user_audio)
            ref_segments = segment_audio_by_silence(ref_audio)
            
            # Align segments with text
            user_aligned = align_segments_with_words(user_segments, text)
            
            # If we have a different number of segments in user and reference, use DTW to align
            if len(user_segments) != len(ref_segments):
                # Keep user segments, but try to align with reference
                word_scores = []
                for (start, end), word in user_aligned:
                    # Extract user segment
                    user_segment = user_audio[start:end]
                    
                    # Find best matching segment in reference
                    best_score = 0
                    for ref_start, ref_end in ref_segments:
                        ref_segment = ref_audio[ref_start:ref_end]
                        score = compute_audio_similarity(user_segment, ref_segment)
                        if score > best_score:
                            best_score = score
                    
                    word_scores.append({
                        "word": word,
                        "status": "Correct" if best_score > 0.5 else "Incorrect"
                    })
            else:
                # Same number of segments, align directly
                word_scores = []
                for i, ((start, end), word) in enumerate(user_aligned):
                    user_segment = user_audio[start:end]
                    ref_start, ref_end = ref_segments[i]
                    ref_segment = ref_audio[ref_start:ref_end]
                    
                    score = compute_audio_similarity(user_segment, ref_segment)
                    
                    word_scores.append({
                        "word": word,
                        "status": "Correct" if score > 0.5 else "Incorrect"
                    })
            
            return {
                "overall": overall_score,
                "word_scores": word_scores,
                "text": text,
                "reference_used": reference_path
            }
    
    # Otherwise use the regular evaluation approach
    # Load the audio
    audio = load_audio(audio_path)
    sr = 16000  # Assuming 16kHz sampling rate
    
    # Segment the audio into potential words
    segments = segment_audio_by_silence(audio, sr)
    
    # Align audio segments with words
    aligned_segments = align_segments_with_words(segments, text)
    
    # Prepare model
    model, processor = prepare_model(model_type)
    
    # Process each word
    word_scores = []
    for (start, end), word in aligned_segments:
        # Extract the audio segment for this word
        word_audio = audio[start:end]
        
        # Skip very short segments
        if len(word_audio) < sr * 0.1:  # Less than 100ms
            continue
            
        # Process this segment
        score = process_audio_segment(word_audio, word, model, processor, model_type)
        
        word_scores.append({
            "word": word,
            "status": "Correct" if score > 0.5 else "Incorrect"
        })
    
    # Also process the entire utterance for comparison
    full_score = check_pronunciation(audio_path, text, model_type)
    
    return {
        "overall": full_score,
        "word_scores": word_scores,
        "text": text
    }


def prepare_model(model_type: str = None):
    """Prepare model and preprocessing function based on model type."""
    # Find available models
    available_models = get_available_models()
    
    # If model_type not specified, use best available model with preference order
    if model_type is None:
        model_preferences = ["wav2vec2", "phoneme", "regression", "mfcc"]
        for pref in model_preferences:
            matching_models = [m for m in available_models if m["type"] == pref]
            if matching_models:
                model_type = pref
                model_path = [m for m in matching_models if "best" in m["path"]][0]["path"] \
                             if any("best" in m["path"] for m in matching_models) \
                             else matching_models[0]["path"]
                break
        else:
            # Fallback to original MFCC model
            model_type = "mfcc"
            model_path = "models/pronunciation_model_best.pt"
    else:
        # Use specified model type
        matching_models = [m for m in available_models if m["type"] == model_type]
        if matching_models:
            model_path = [m for m in matching_models if "best" in m["path"]][0]["path"] \
                         if any("best" in m["path"] for m in matching_models) \
                         else matching_models[0]["path"]
        else:
            # Fallback to the original model
            model_path = "models/pronunciation_model_best.pt"
            model_type = "mfcc"
    
    # Load the model
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint.get('config', {'model_type': model_type})
        
        # Create appropriate model based on type
        if model_type == "wav2vec2" or config.get('model_type') == "wav2vec2":
            model = TransferLearningPronunciationClassifier()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            def processor(audio, text=None):
                # Convert to tensor
                waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                
                # Normalize
                waveform = waveform / (waveform.abs().max() + 1e-7)
                
                return waveform, None
                
        elif model_type == "phoneme" or config.get('model_type') == "phoneme":
            model = PhonemeEmbeddingPronunciationClassifier(
                phoneme_vocab_size=len(PHONEME_TO_IDX) if PHONEME_UTILS_AVAILABLE else 50
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            def processor(audio, text=None):
                # Process audio to MFCC
                mfcc = extract_mfcc(audio)
                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
                
                # Process text to phonemes if available
                phoneme_tensor = None
                if PHONEME_UTILS_AVAILABLE and text:
                    phonemes = text_to_phonemes(text)
                    indices = [PHONEME_TO_IDX.get(ph, PHONEME_TO_IDX['<UNK>']) for ph in phonemes]
                    
                    # Pad or truncate phoneme sequence
                    max_phoneme_len = 100
                    if len(indices) > max_phoneme_len:
                        indices = indices[:max_phoneme_len]
                    else:
                        indices = indices + [PHONEME_TO_IDX['<PAD>']] * (max_phoneme_len - len(indices))
                        
                    phoneme_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
                
                return mfcc_tensor, phoneme_tensor
                
        elif model_type == "regression" or config.get('model_type') == "regression":
            model = RegressionPronunciationClassifier()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            def processor(audio, text=None):
                # Process audio to MFCC
                mfcc = extract_mfcc(audio)
                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
                return mfcc_tensor, None
                
        else:  # Default MFCC model
            model = PronunciationClassifier()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            def processor(audio, text=None):
                # Process audio to MFCC
                mfcc = extract_mfcc(audio)
                mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
                return mfcc_tensor, None
            
        model.eval()
        return model, processor
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to original MFCC model
        model = PronunciationClassifier()
        
        try:
            model.load_state_dict(torch.load("models/pronunciation_model_best.pt", map_location=torch.device('cpu')))
        except:
            # Last resort: just use the model without loading weights
            pass
            
        def processor(audio, text=None):
            # Process audio to MFCC
            mfcc = extract_mfcc(audio)
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            return mfcc_tensor, None
            
        model.eval()
        return model, processor


def process_audio_segment(audio_segment, word, model, processor, model_type):
    """Process a single audio segment with the model."""
    # Preprocess the audio segment
    main_input, aux_input = processor(audio_segment, word)
    
    # Run through model
    with torch.no_grad():
        if aux_input is not None:
            # Model with auxiliary input (like phoneme model)
            output = model(main_input, aux_input).item()
        else:
            # Models with single input
            output = model(main_input).item()
            
        # Handle regression models
        if model_type == "regression":
            # Clamp to [0, 1] range
            output = max(0, min(1, output))
            
    return output


def check_pronunciation(audio_path: str, text: str = None, model_type: str = None) -> dict:
    """
    Full pipeline to check pronunciation from an audio file.
    Returns a confidence score and classification.
    
    Args:
        audio_path: Path to audio file
        text: Optional transcription text (required for phoneme-based models)
        model_type: Optional model type to use (wav2vec2, mfcc, phoneme, regression)
    """
    # Find available models
    available_models = get_available_models()
    
    # If model_type not specified, use best available model with preference order:
    # 1. wav2vec2, 2. phoneme, 3. regression, 4. mfcc
    if model_type is None:
        model_preferences = ["wav2vec2", "phoneme", "regression", "mfcc"]
        for pref in model_preferences:
            matching_models = [m for m in available_models if m["type"] == pref]
            if matching_models:
                model_type = pref
                model_path = [m for m in matching_models if "best" in m["path"]][0]["path"] \
                             if any("best" in m["path"] for m in matching_models) \
                             else matching_models[0]["path"]
                break
        else:
            # Fallback to original MFCC model
            model_type = "mfcc"
            model_path = "models/pronunciation_model_best.pt"
    else:
        # Use specified model type
        matching_models = [m for m in available_models if m["type"] == model_type]
        if matching_models:
            model_path = [m for m in matching_models if "best" in m["path"]][0]["path"] \
                         if any("best" in m["path"] for m in matching_models) \
                         else matching_models[0]["path"]
        else:
            # Fallback to the original model
            model_path = "models/pronunciation_model_best.pt"
            model_type = "mfcc"
    
    print(f"Using model type: {model_type}, path: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint.get('config', {'model_type': model_type})
        
        # Process based on model type
        if model_type == "wav2vec2" or config.get('model_type') == "wav2vec2":
            # Using wav2vec2 model
            model = TransferLearningPronunciationClassifier()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Process audio for wav2vec2
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
            
            # Normalize
            waveform = waveform / (waveform.abs().max() + 1e-7)
            
            # Prepare input tensor
            input_tensor = waveform.squeeze(0).unsqueeze(0)  # [1, time]
            
            # No additional inputs needed
            model_inputs = [input_tensor]
            
        elif model_type == "phoneme" or config.get('model_type') == "phoneme":
            # Using phoneme embedding model
            if not text:
                raise ValueError("Text transcription is required for phoneme-based models")
                
            if not PHONEME_UTILS_AVAILABLE:
                raise ImportError("Phoneme utilities are not available")
                
            # Initialize model
            model = PhonemeEmbeddingPronunciationClassifier(
                phoneme_vocab_size=len(PHONEME_TO_IDX),
                phoneme_embed_dim=64
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Process audio
            audio = load_audio(audio_path)
            mfcc = extract_mfcc(audio)
            audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            
            # Process text to phonemes
            phonemes = text_to_phonemes(text)
            indices = [PHONEME_TO_IDX.get(ph, PHONEME_TO_IDX['<UNK>']) for ph in phonemes]
            
            # Pad or truncate phoneme sequence
            max_phoneme_len = 100
            if len(indices) > max_phoneme_len:
                indices = indices[:max_phoneme_len]
            else:
                indices = indices + [PHONEME_TO_IDX['<PAD>']] * (max_phoneme_len - len(indices))
                
            phoneme_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
            
            # Prepare inputs
            model_inputs = [audio_tensor, phoneme_tensor]
            
        elif model_type == "regression" or config.get('model_type') == "regression":
            # Using regression model
            model = RegressionPronunciationClassifier()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Process audio
            audio = load_audio(audio_path)
            mfcc = extract_mfcc(audio)
            input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            
            # No additional inputs needed
            model_inputs = [input_tensor]
            
        else:
            # Using original MFCC model
            model = PronunciationClassifier()
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Process audio
            audio = load_audio(audio_path)
            mfcc = extract_mfcc(audio)
            input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
            
            # No additional inputs needed
            model_inputs = [input_tensor]
    
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to original MFCC model
        model = PronunciationClassifier()
        model_path = "models/pronunciation_model_best.pt"
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except:
            # Last resort: load the model without state dict
            pass
        
        # Process audio for MFCC model
        audio = load_audio(audio_path)
        mfcc = extract_mfcc(audio)
        input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
        
        # No additional inputs needed
        model_inputs = [input_tensor]

    # Evaluate
    model.eval()
    with torch.no_grad():
        if len(model_inputs) == 1:
            score = model(model_inputs[0]).item()
        else:
            score = model(*model_inputs).item()
            
        # Handle regression models (they don't have sigmoid activation)
        if model_type == "regression" or config.get('model_type') == "regression":
            # Clamp score to [0, 1] range
            score = max(0, min(1, score))

    # Return simplified binary classification output
    return {
        "status": "Correct" if score > 0.5 else "Incorrect",
        "model_type": model_type
    }
