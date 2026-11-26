import os

os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

import random
import numpy as np
import torch
from dnn import (
    BASE_LIBRISPEECH_DIR,
    raw_ds,
    LibriSpeechFramesDataset,
    NUM_CLASSES,
    STATE_TO_ID,
    SpeechDNN,
    process_utterance,
)
from hmm import FullWordViterbiDecoder


def run_word_evaluation(num_samples=3, num_prior_samples=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize the dataset directly in this script
    train_dataset = LibriSpeechFramesDataset(
        raw_ds, BASE_LIBRISPEECH_DIR, max_utterances=22000, device=device
    )

    # 2. Instantiate the Uniform 5-Layer DNN
    model = SpeechDNN(
        input_dim=880, hidden_dim=1024, output_dim=NUM_CLASSES
    ).to(device)

    # 3. Load the best saved checkpoint
    weights_path = "sparse_speech_dnn_weights.pth"
    if not os.path.exists(weights_path):
        print(f"[ERROR]: Could not find weights checkpoint at '{weights_path}'")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"Loaded checkpoint weights from: {weights_path}")

    # 4. Verify the 3-state expanded lexicon exists
    lexicon_file = "expanded_lexicon_3state.txt"
    if not os.path.exists(lexicon_file):
        print(
            f"[ERROR]: '{lexicon_file}' not found. Run your 3-state lexicon builder script first."
        )
        return

    # Pre-load global 3-state lexicon into memory
    global_lexicon = {}
    with open(lexicon_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                global_lexicon[parts[0]] = parts[1:]

    # =====================================================================
    # 5. COMPUTE CLASS PRIORS FROM THE DATASET
    # =====================================================================
    print(f"\nComputing class priors from {num_prior_samples} utterances...")
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)
    
    # We sample a subset to estimate priors quickly without iterating the whole 22k dataset
    prior_indices = random.sample(range(len(train_dataset)), min(num_prior_samples, len(train_dataset)))
    
    for idx in prior_indices:
        utterance_id = train_dataset.ids[idx]
        phoneme_intervals = train_dataset.phonemes_data[idx]
        spk_id, chap_id, _ = utterance_id.split("-")
        audio_path = os.path.join(
            train_dataset.base_dir, "train-clean-100", spk_id, chap_id, f"{utterance_id}.flac"
        )
        
        try:
            _, targets = process_utterance(audio_path, phoneme_intervals, device)
            # Count occurrences of each class ID in the target frames
            unique_classes, counts = torch.unique(targets, return_counts=True)
            class_counts[unique_classes] += counts.float()
        except Exception:
            continue

    # Add a small epsilon to prevent log(0) / division by zero later
    class_counts += 1e-5
    class_priors = (class_counts / class_counts.sum()).cpu().numpy()
    print("Class priors computation complete.")
    # =====================================================================

    # Pick random utterances from the dataset index for the actual evaluation
    indices_to_test = random.sample(range(len(train_dataset)), num_samples)

    print("\n" + "=" * 70)
    print(" 3-STATE DUAL-MODULE DYNAMIC TARGET VITERBI EVALUATION")
    print("=" * 70)

    with torch.no_grad():
        for sample_no, idx in enumerate(indices_to_test, 1):
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

            # Fetch the ground truth text transcript for this utterance
            trans_dir = os.path.join(
                train_dataset.base_dir, "train-clean-100", spk_id, chap_id
            )
            trans_file = os.path.join(
                trans_dir, f"{spk_id}-{chap_id}.trans.txt"
            )

            ground_truth_words = []
            if os.path.exists(trans_file):
                with open(trans_file, "r") as tf:
                    for line in tf:
                        if line.startswith(utterance_id):
                            ground_truth_words = line.strip().split()[1:]
                            break

            if not ground_truth_words:
                print(
                    f"Skipping sample {utterance_id}: Transcript text not found."
                )
                continue

            # Build a dynamic sentence-level local lexicon using 3-state sequences
            target_vocab = set(ground_truth_words)
            local_lexicon_path = "temp_local_lexicon.txt"

            with open(local_lexicon_path, "w", encoding="utf-8") as lf:
                for word in target_vocab:
                    if word in global_lexicon:
                        lf.write(f"{word} {' '.join(global_lexicon[word])}\n")

            try:
                # Instantiate 3-State Viterbi Decoder with STATE_TO_ID and class_priors
                word_decoder = FullWordViterbiDecoder(
                    lexicon_path=local_lexicon_path, 
                    state_map=STATE_TO_ID,
                    class_priors=class_priors  # PASSED THE ESTIMATED PRIORS HERE
                )

                inputs, targets = process_utterance(
                    audio_path, phoneme_intervals, device
                )
                logits = model(inputs)
                posteriors = torch.softmax(logits, dim=1).cpu().numpy()

                # Decode using tuned 3-state HMM parameters
                reconstructed_text = word_decoder.decode(
                    posteriors,
                    word_insertion_penalty=1.5,
                    acoustic_scale=0.17,
                )

                print(f"\n[Sample {sample_no}] Utterance ID: {utterance_id}")
                print(f"📖 GROUND TRUTH:  {' '.join(ground_truth_words)}")
                print(f"🔮 RECONSTRUCTED: {reconstructed_text}")
                print("-" * 50)

            except Exception as e:
                print(
                    f"Skipping sample {utterance_id} due to evaluation error: {e}"
                )

            finally:
                if os.path.exists(local_lexicon_path):
                    os.remove(local_lexicon_path)


if __name__ == "__main__":
    # You can increase num_prior_samples for a more accurate prior distribution
    # 200 utterances is generally enough for a stable baseline
    run_word_evaluation(num_samples=3, num_prior_samples=200)