import os
import torch
import torch.nn as nn
import torch.optim as optim
from block_pruner import BlockPruner
from dnn import (
    BASE_LIBRISPEECH_DIR,
    NUM_CLASSES,
    STATE_TO_ID,
    LibriSpeechFramesDataset,
    SpeechDNN,
    process_utterance,
    raw_ds,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Instantiate the model
model = SpeechDNN(
    input_dim=880, hidden_dim=1024, output_dim=NUM_CLASSES
).to(device)

# 2. Resume from the SPARSE checkpoint, NOT the dense one
sparse_weights_path = "sparse_speech_dnn_weights.pth"
masks_path = "sparse_block_masks.pth"

if os.path.exists(sparse_weights_path) and os.path.exists(masks_path):
    print(f"Resuming fine-tuning from {sparse_weights_path}...")
    model.load_state_dict(torch.load(sparse_weights_path, map_location=device))
    saved_masks = torch.load(masks_path, map_location=device)
else:
    raise FileNotFoundError(
        "Missing sparse_speech_dnn_weights.pth or sparse_block_masks.pth!"
    )

# 3. Load Dataset
train_dataset = LibriSpeechFramesDataset(
    raw_ds, BASE_LIBRISPEECH_DIR, max_utterances=15000, device=device
)

# 4. Attach the EXISTING masks without recalculating
pruner = BlockPruner(model, block_size=64, keep_ratio=0.25)
pruner.masks = saved_masks  # Overwrite with the original 75% mask
pruner.apply_mask()

# 5. Set a slightly lower LR for second-stage convergence
optimizer = optim.Adam(model.parameters(), lr=0.00005)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=1
)

best_sparse_acc = 63.33  # Baseline from previous run
NUM_EPOCHS = 5

print(
    f"\nStarting Second-Stage Fine-Tuning ({NUM_EPOCHS} epochs, LR=5e-5)..."
)

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    processed = 0

    shuffled_indices = torch.randperm(len(train_dataset)).tolist()

    for idx in shuffled_indices:
        utterance_id = train_dataset.ids[idx]
        phoneme_intervals = train_dataset.phonemes_data[idx]
        spk_id, chap_id, _ = utterance_id.split("-")
        audio_path = os.path.join(
            train_dataset.base_dir,
            "train-clean-100",
            spk_id,
            chap_id,
            f"{utterance_id}.flac",
        )

        try:
            inputs, targets = process_utterance(
                audio_path, phoneme_intervals, device
            )
        except Exception:
            continue

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Enforce zero gradients on pruned blocks
        pruner.zero_masked_gradients()

        optimizer.step()

        # Ensure exact zeros on pruned weights
        pruner.apply_mask()

        _, predicted = torch.max(outputs.data, 1)
        epoch_total += targets.size(0)
        epoch_correct += (predicted == targets).sum().item()
        epoch_loss += loss.item()
        processed += 1

        if processed % 100 == 0:
            acc = (epoch_correct / epoch_total) * 100
            print(
                f"Epoch {epoch} | Processed: {processed}/{len(train_dataset)} | Running Acc: {acc:.2f}%"
            )

    # Validation Phase
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for val_idx in shuffled_indices[:100]:
            v_utt = train_dataset.ids[val_idx]
            v_intervals = train_dataset.phonemes_data[val_idx]
            s_id, c_id, _ = v_utt.split("-")
            a_path = os.path.join(
                train_dataset.base_dir,
                "train-clean-100",
                s_id,
                c_id,
                f"{v_utt}.flac",
            )
            try:
                v_in, v_tgt = process_utterance(a_path, v_intervals, device)
                v_out = model(v_in)
                _, v_pred = torch.max(v_out.data, 1)
                val_total += v_tgt.size(0)
                val_correct += (v_pred == v_tgt).sum().item()
            except Exception:
                continue

    val_acc = (val_correct / val_total) * 100
    print(f"\n=== Epoch {epoch} Evaluation ===")
    print(f"--> Sparse Model Frame Accuracy: {val_acc:.2f}%")

    scheduler.step(val_acc)
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

    if val_acc > best_sparse_acc:
        best_sparse_acc = val_acc
        torch.save(model.state_dict(), "sparse_speech_dnn_weights.pth")
        torch.save(pruner.masks, "sparse_block_masks.pth")
        print(
            f"New best model saved to 'sparse_speech_dnn_weights.pth' ({best_sparse_acc:.2f}%)\n"
        )
    else:
        print(f"No improvement over {best_sparse_acc:.2f}%. Not overwritten.\n")