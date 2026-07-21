import os

os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import soundfile as sf
from datasets import load_dataset


# ==========================================
# 1. CONFIGURATION & DATASET PATHS
# ==========================================
# UPDATE THIS PATH to point to your actual local LibriSpeech folder directory
BASE_LIBRISPEECH_DIR = r"D:\SpeechRecognitionDDNinVHDL\Python\data\LibriSpeech ASR corpus\LibriSpeech" 

print("Loading training alignment metadata from Hugging Face...")
# Load the dataset but immediately strip any audio decoding dependencies by converting to a raw dict format
raw_ds = load_dataset("gilkeyio/librispeech-alignments", split="train_clean_100")

print("Extracting unique phoneme set...")
# Iterating directly through the columns without invoking Hugging Face's custom format row decoders
UNIQUE_PHONEMES = sorted(list(set(
    p['phoneme'] 
    for phonemes_list in raw_ds['phonemes'] 
    for p in phonemes_list 
    if 'phoneme' in p
)))

PHONEME_TO_ID = {phone: idx for idx, phone in enumerate(UNIQUE_PHONEMES)}
PHONEME_TO_ID["SIL"] = len(PHONEME_TO_ID) 

NUM_CLASSES = len(PHONEME_TO_ID)
print(f"Total Unique Phoneme Targets Found: {NUM_CLASSES}")

# ==========================================
# 2. FEATURE EXTRACTION & SPLICING ENGINE
# ==========================================
def process_utterance(audio_path, phoneme_intervals, device, context_window=5):
    # 1. Read file on CPU (soundfile must read to CPU memory first)
    data, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
    # 2. IMMEDIATELY move the raw waveform to the GPU
    waveform = waveform.to(device)
    
    # 3. Perform Mel Spectrogram directly on GPU fabric
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=40
    ).to(device)
    
    mel_spec = torch.log(mel_transform(waveform) + 1e-9).squeeze(0).T
    num_frames = mel_spec.shape[0]
    
    # 4. Target Alignment (Calculated on CPU/GPU seamlessly)
    frame_labels = []
    for frame_idx in range(num_frames):
        current_time_sec = frame_idx * 0.01
        active_phone = "SIL"
        for p in phoneme_intervals:
            if p['start'] <= current_time_sec <= p['end']:
                active_phone = p['phoneme']
                break
        frame_labels.append(PHONEME_TO_ID[active_phone])
    
    # 5. Context Splicing on GPU
    padded_mel = torch.nn.functional.pad(
        mel_spec, (0, 0, context_window, context_window), mode='constant', value=0
    )
    spliced_frames = []
    for t in range(num_frames):
        window = padded_mel[t : t + (2 * context_window) + 1]
        spliced_frames.append(window.flatten())
        
    return torch.stack(spliced_frames), torch.tensor(frame_labels, dtype=torch.long).to(device)

# ==========================================
# 3. PYTORCH DATASET DATA LOADER
# ==========================================
class LibriSpeechFramesDataset(Dataset):
    def __init__(self, alignments_subset, base_dir, max_utterances=1000):
        self.base_dir = base_dir
        
        # Limit how many files we look at (start with 1,000 utterances)
        num_utterances = min(max_utterances, len(alignments_subset))
        
        # Keep only the metadata pointers in memory (uses almost zero RAM)
        self.ids = alignments_subset['id'][:num_utterances]
        self.phonemes_data = alignments_subset['phonemes'][:num_utterances]
        
        print(f"Dataset initialized with {num_utterances} utterances. Features will load dynamically!")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        utterance_id = self.ids[idx]
        spk_id, chap_id, _ = utterance_id.split("-")
        audio_path = f"{self.base_dir}\\dev-clean\\{spk_id}\\{chap_id}\\{utterance_id}.flac"
        
        try:
            # Load and process just this ONE file when requested
            X_utt, y_utt = process_utterance(audio_path, self.phonemes_data[idx])
            return X_utt, y_utt
        except Exception:
            # Fallback placeholder in case a file is corrupted/missing
            return torch.zeros((1, 440)), torch.zeros((1,), dtype=torch.long)

# ==========================================
# 4. DNN MODEL ARCHITECTURE
# ==========================================
class SpeechDNN(nn.Module):
    def __init__(self, input_dim=440, hidden_dim_1=1024, hidden_dim_2=2048, output_dim=71):
        super(SpeechDNN, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: Expand to baseline hidden size
            nn.Linear(input_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Layer 2: Expand to the peak of the diamond (2048 neurons)
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Layer 3: Compress back down to hidden baseline (1024 neurons)
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Output Layer
            nn.Linear(hidden_dim_1, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 5. INITIALIZATION AND TRAINING LOOP
# ==========================================
# 1. Initialize dataset
train_dataset = LibriSpeechFramesDataset(raw_ds, BASE_LIBRISPEECH_DIR, max_utterances=15000)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We use num_workers=0 or 1 on Windows here because we are reading metadata directly 
    # from the dataset object via the loop index.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    model = SpeechDNN(input_dim=440, hidden_dim_1=1024, hidden_dim_2=2048, output_dim=NUM_CLASSES).to(device)

    # --- RESUME PROGRESS ---
    import os
    if os.path.exists("speech_dnn_weights.pth"):
        print("Found existing weights! Loading progress...")
        model.load_state_dict(torch.load("speech_dnn_weights.pth", map_location=device))
    else:
        print("No saved weights found. Starting training from scratch with random values.")
    # -----------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Add scheduler to smoothly lower the learning rate when accuracy plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    NUM_EPOCHS = 10 
    
    print(f"\nStarting scalable training loop for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train() # Make sure model is in training mode at the start of each epoch
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        processed_utterances = 0
        
        # Shuffle indices manually so we can access raw_ds directly and keep shuffle functionality
        shuffled_indices = torch.randperm(len(train_dataset)).tolist()
        
        for idx in shuffled_indices:
            utterance_id = train_dataset.ids[idx]
            phoneme_intervals = train_dataset.phonemes_data[idx]
            
            spk_id, chap_id, _ = utterance_id.split("-")
            
            # --- DOUBLE CHECK THIS PATH MATCHES YOUR WINDOWS EXPLORER FOLDER NAME ---
            audio_path = f"{train_dataset.base_dir}\\train-clean-100\\{spk_id}\\{chap_id}\\{utterance_id}.flac"
            
            try:
                # Process features directly on whichever device is active (CPU or GPU)
                inputs, targets = process_utterance(audio_path, phoneme_intervals, device)
            except Exception as e:
                # If a file fails, print out why so we aren't flying blind!
                if processed_utterances == 0:
                    print(f"[DEBUG Error] Skipping file because: {e}")
                    print(f"[DEBUG Path tried]: {audio_path}")
                continue
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
            epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            processed_utterances += 1
            
            if processed_utterances % 50 == 0:
                current_acc = (epoch_correct / epoch_total) * 100
                print(f"Epoch {epoch} | Utterances: {processed_utterances}/{len(train_dataset)} | Running Acc: {current_acc:.2f}%")
        
        # Prevent zero division crash if the entire dataset failed to load (Checked AFTER the loop finishes)
        if processed_utterances == 0:
            print("\n[ERROR]: Zero utterances were processed. Check your dataset paths above!")
            break
                
        epoch_final_acc = (epoch_correct / epoch_total) * 100
        epoch_avg_loss = epoch_loss / processed_utterances
        
        print(f"\n=== Epoch {epoch} Complete ===")
        print(f"Training Running Accuracy (with Dropout): {epoch_final_acc:.2f}%")
        print(f"Average Loss: {epoch_avg_loss:.4f}")
        
        # ====================================================
        # EVALUATION ENGINE
        # ====================================================
        model.eval()  # Temporarily turns off Dropout and BatchNorm training adjustments
        
        eval_correct = 0
        eval_total = 0
        
        # Test on a small subset of 50 utterances with full network power
        print("Calculating True Model Accuracy with all neurons active...")
        with torch.no_grad():
            for val_idx in shuffled_indices[:50]: 
                v_utterance_id = train_dataset.ids[val_idx]
                v_phoneme_intervals = train_dataset.phonemes_data[val_idx]
                spk_id, chap_id, _ = v_utterance_id.split("-")
                v_audio_path = f"{train_dataset.base_dir}\\train-clean-100\\{spk_id}\\{chap_id}\\{v_utterance_id}.flac"
                
                try:
                    v_inputs, v_targets = process_utterance(v_audio_path, v_phoneme_intervals, device)
                    v_outputs = model(v_inputs)
                    _, v_predicted = torch.max(v_outputs.data, 1)
                    eval_total += v_targets.size(0)
                    eval_correct += (v_predicted == v_targets).sum().item()
                except Exception:
                    continue
                    
        true_accuracy = (eval_correct / eval_total) * 100
        print(f"--> True Model Accuracy (All Neurons Active): {true_accuracy:.2f}%")
        # ====================================================
        
        # Adjust learning rate based on true performance (only step once based on validation)
        scheduler.step(true_accuracy)
        print(f"Learning Rate for next epoch: {optimizer.param_groups[0]['lr']}\n")
        
        # --- SAVE CHECKPOINT AT END OF EVERY EPOCH ---
        torch.save(model.state_dict(), "speech_dnn_weights.pth")
        print("Progress checkpoint saved to speech_dnn_weights.pth\n")
        torch.save(model.state_dict(), "speech_dnn_weights.pth")
        print("Model weights successfully saved to speech_dnn_weights.pth!")