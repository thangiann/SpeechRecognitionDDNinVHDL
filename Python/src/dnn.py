import os

os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.functional as F
import torchaudio.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. CONFIGURATION & DATASET PATHS
# ==========================================
BASE_LIBRISPEECH_DIR = r"D:\SpeechRecognitionDDNinVHDL\Python\data\LibriSpeech ASR corpus\LibriSpeech"

print("Loading training alignment metadata from Hugging Face...")
raw_ds = load_dataset("gilkeyio/librispeech-alignments", split="train_clean_100")

print("Extracting unique phoneme set...")
UNIQUE_PHONEMES = sorted(
    list(
        set(
            p["phoneme"]
            for phonemes_list in raw_ds["phonemes"]
            for p in phonemes_list
            if "phoneme" in p
        )
    )
)

# Build 3-State Target mapping
STATE_TO_ID = {}
curr_idx = 0
for phone in UNIQUE_PHONEMES:
    for state in range(3):
        STATE_TO_ID[f"{phone}_{state}"] = curr_idx
        curr_idx += 1

STATE_TO_ID["SIL"] = curr_idx
NUM_CLASSES = len(STATE_TO_ID)
print(f"Base Phonemes: {len(UNIQUE_PHONEMES)} | Total Output Targets: {NUM_CLASSES}")


# ==========================================
# 2. FEATURE EXTRACTION (STATIC + DELTA, W=5)
# ==========================================
def process_utterance(audio_path, phoneme_intervals, device, context_window=5):
    data, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    # 1. Base Log-Mel Spectrogram (40 mels)
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=40
    ).to(device)

    mel_spec = torch.log(mel_transform(waveform) + 1e-9).squeeze(0)  # Shape: [40, Time]

    # 2. Compute Delta Features ONLY (Drop Delta-Delta)
    delta1 = F.compute_deltas(mel_spec)

    # Stack static + delta -> Shape: [80, Time]
    full_features = torch.cat([mel_spec, delta1], dim=0)

    # 3. Cepstral Mean & Variance Normalization (CMVN)
    mean = full_features.mean(dim=1, keepdim=True)
    std = full_features.std(dim=1, keepdim=True) + 1e-5
    norm_features = (full_features - mean) / std

    # Transpose to [Time, 80]
    norm_features = norm_features.T
    num_frames = norm_features.shape[0]

    # 4. Target Alignment (3-State Mapping)
    frame_labels = []
    for frame_idx in range(num_frames):
        current_time_sec = frame_idx * 0.01
        active_phone = "SIL"
        state_idx = 0

        for p in phoneme_intervals:
            if p["start"] <= current_time_sec <= p["end"]:
                active_phone = p["phoneme"]
                duration = p["end"] - p["start"]

                if duration > 0:
                    num_p_frames = int(duration / 0.01)
                    if num_p_frames < 3:
                        # Short phonemes stick to the middle steady state (S1)
                        state_idx = 1
                    else:
                        rel_pos = (current_time_sec - p["start"]) / duration
                        if rel_pos < 0.33:
                            state_idx = 0
                        elif rel_pos < 0.66:
                            state_idx = 1
                        else:
                            state_idx = 2
                break

        target_key = "SIL" if active_phone == "SIL" else f"{active_phone}_{state_idx}"
        frame_labels.append(STATE_TO_ID[target_key])

    # 5. Context Splicing ([Time, 80 * (2 * 5 + 1)] = [Time, 880])
    padded_mel = torch.nn.functional.pad(
        norm_features,
        (0, 0, context_window, context_window),
        mode="constant",
        value=0,
    )
    spliced_frames = [
        padded_mel[t : t + (2 * context_window) + 1].flatten()
        for t in range(num_frames)
    ]

    return torch.stack(spliced_frames), torch.tensor(
        frame_labels, dtype=torch.long
    ).to(device)


# ==========================================
# 3. DATASET DATA LOADER
# ==========================================
class LibriSpeechFramesDataset(Dataset):

    def __init__(
        self,
        alignments_subset,
        base_dir,
        max_utterances=1000,
        device=torch.device("cpu"),
    ):
        self.base_dir = base_dir
        self.device = device
        num_utterances = min(max_utterances, len(alignments_subset))
        self.ids = alignments_subset["id"][:num_utterances]
        self.phonemes_data = alignments_subset["phonemes"][:num_utterances]

        print(f"Dataset initialized with {num_utterances} utterances.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        utterance_id = self.ids[idx]
        spk_id, chap_id, _ = utterance_id.split("-")
        audio_path = f"{self.base_dir}\\train-clean-100\\{spk_id}\\{chap_id}\\{utterance_id}.flac"

        try:
            X_utt, y_utt = process_utterance(
                audio_path, self.phonemes_data[idx], self.device
            )
            return X_utt, y_utt
        except Exception:
            return torch.zeros((1, 880)), torch.zeros((1,), dtype=torch.long)


# ==========================================
# 4. DNN MODEL ARCHITECTURE (UNIFORM 5-LAYER)
# ==========================================
class SpeechDNN(nn.Module):
    def __init__(self, input_dim=880, hidden_dim=1024, output_dim=213):
        super(SpeechDNN, self).__init__()
        self.network = nn.Sequential(
            # Layer 1: Input Projection
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Layer 2: Hidden 1
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Layer 3: Hidden 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Layer 4: Hidden 3
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Layer 5: Hidden 4
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            # Layer 6: Output Classifier
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
# ==========================================
# 5. INITIALIZATION AND TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = LibriSpeechFramesDataset(
        raw_ds, BASE_LIBRISPEECH_DIR, max_utterances=22000, device=device
    )

    model = SpeechDNN(
        input_dim=880, hidden_dim=1024, output_dim=NUM_CLASSES
    ).to(device)

    best_val_acc = 0.0  # Track top performance

    if os.path.exists("speech_dnn_weights.pth"):
        print("Found existing weights! Resuming progress...")
        try:
            model.load_state_dict(
                torch.load("speech_dnn_weights.pth", map_location=device)
            )
            print("Successfully loaded existing checkpoint.")
        except Exception as e:
            print(f"Resetting weights due to shape mismatch: {e}")
            os.remove("speech_dnn_weights.pth")

    criterion = nn.CrossEntropyLoss()
    # Lower initial LR slightly for deep 5-layer stability
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Trigger LR reduction with patience=1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-5
    )

    NUM_EPOCHS = 15
    print(
        f"\nStarting training loop ({NUM_EPOCHS} epochs) with best-checkpoint tracking..."
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        processed_utterances = 0

        shuffled_indices = torch.randperm(len(train_dataset)).tolist()

        for idx in shuffled_indices:
            utterance_id = train_dataset.ids[idx]
            phoneme_intervals = train_dataset.phonemes_data[idx]
            spk_id, chap_id, _ = utterance_id.split("-")
            audio_path = f"{train_dataset.base_dir}\\train-clean-100\\{spk_id}\\{chap_id}\\{utterance_id}.flac"

            try:
                inputs, targets = process_utterance(
                    audio_path, phoneme_intervals, device
                )
            except Exception:
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

            if processed_utterances % 100 == 0:
                current_acc = (epoch_correct / epoch_total) * 100
                print(
                    f"Epoch {epoch} | Utterances: {processed_utterances}/{len(train_dataset)} | Running Acc: {current_acc:.2f}%"
                )

        if processed_utterances == 0:
            print("\n[ERROR]: Zero utterances processed. Check audio path!")
            break

        epoch_final_acc = (epoch_correct / epoch_total) * 100
        epoch_avg_loss = epoch_loss / processed_utterances

        print(f"\n=== Epoch {epoch} Complete ===")
        print(f"Training Running Accuracy: {epoch_final_acc:.2f}%")
        print(f"Average Loss: {epoch_avg_loss:.4f}")

        # Evaluation Phase
        model.eval()
        eval_correct = 0
        eval_total = 0

        with torch.no_grad():
            for val_idx in shuffled_indices[:100]:
                v_utterance_id = train_dataset.ids[val_idx]
                v_phoneme_intervals = train_dataset.phonemes_data[val_idx]
                spk_id, chap_id, _ = v_utterance_id.split("-")
                v_audio_path = f"{train_dataset.base_dir}\\train-clean-100\\{spk_id}\\{chap_id}\\{v_utterance_id}.flac"

                try:
                    v_inputs, v_targets = process_utterance(
                        v_audio_path, v_phoneme_intervals, device
                    )
                    v_outputs = model(v_inputs)
                    _, v_predicted = torch.max(v_outputs.data, 1)
                    eval_total += v_targets.size(0)
                    eval_correct += (v_predicted == v_targets).sum().item()
                except Exception:
                    continue

        true_accuracy = (eval_correct / eval_total) * 100
        print(f"--> True Model Accuracy: {true_accuracy:.2f}%")

        scheduler.step(true_accuracy)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate for next epoch: {current_lr}")

        # ONLY save checkpoint if performance improved
        if true_accuracy > best_val_acc:
            best_val_acc = true_accuracy
            torch.save(model.state_dict(), "speech_dnn_weights.pth")
            print(
                f"🔥 New best model found ({best_val_acc:.2f}%)! Saved to speech_dnn_weights.pth\n"
            )
        else:
            print(
                f"Accuracy did not improve from best ({best_val_acc:.2f}%). Checkpoint not overwritten.\n"
            )