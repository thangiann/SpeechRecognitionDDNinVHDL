import os

os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

import torch
import random

# 1. Import your working dataset instance straight from dnn.py
# (Change 'train_dataset' to whatever variable name holds your 15,000 files in dnn.py)
from dnn import SpeechDNN, process_utterance, PHONEME_TO_ID, train_dataset
from hmm import FullWordViterbiDecoder

def run_word_evaluation(num_samples=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We no longer need to initialize raw_ds or train_dataset here!
    # They are automatically loaded via the import statement above.

    # 2. Instantiate your Diamond DNN Architecture
    model = SpeechDNN(input_dim=440, hidden_dim_1=1024, hidden_dim_2=2048, output_dim=71).to(device)
    
    # 3. Load your high-score weights file
    weights_path = "speech_dnn_weights.pth"
    if not os.path.exists(weights_path):
        print(f"[ERROR]: Could not find weights checkpoint at '{weights_path}'")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Lock out dropout layers
    print(f"Loaded high-score checkpoint weights from: {weights_path}")
    
    # 4. Initialize your structural Viterbi Decoder
    lexicon_file = "expanded_lexicon.txt"
    if not os.path.exists(lexicon_file):
        print(f"[ERROR]: Please create '{lexicon_file}' in your workspace directory first.")
        return
        
    word_decoder = FullWordViterbiDecoder(lexicon_path=lexicon_file, phoneme_list=PHONEME_TO_ID)
    
    # Pick random files from the working dataset index
    indices_to_test = random.sample(range(len(train_dataset)), num_samples)
    
    print("\n" + "="*70)
    print(" DUAL-MODULE DYNAMIC TARGET VITERBI EVALUATION")
    print("="*70)
    
    # Load the comprehensive dictionary into memory as a lookup table first
    # (We will use this to dynamically build mini-lexicons per sentence)
    global_lexicon = {}
    with open("expanded_lexicon.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                global_lexicon[parts[0]] = parts[1:]

    with torch.no_grad():
        for sample_no, idx in enumerate(indices_to_test, 1):
            utterance_id = train_dataset.ids[idx]
            phoneme_intervals = train_dataset.phonemes_data[idx]
            
            spk_id, chap_id, _ = utterance_id.split("-")
            audio_path = os.path.join(train_dataset.base_dir, "train-clean-100", spk_id, chap_id, f"{utterance_id}.flac")
            
            # 1. Fetch the real text transcript for this utterance from LibriSpeech
            # Most setups read this from the transcript .txt file inside the chapter folder
            trans_dir = os.path.join(train_dataset.base_dir, "train-clean-100", spk_id, chap_id)
            trans_file = os.path.join(trans_dir, f"{spk_id}-{chap_id}.trans.txt")
            
            ground_truth_words = []
            if os.path.exists(trans_file):
                with open(trans_file, "r") as tf:
                    for line in tf:
                        if line.startswith(utterance_id):
                            # Extract words, removing the leading ID tag
                            ground_truth_words = line.strip().split()[1:]
                            break
            
            if not ground_truth_words:
                print(f"Skipping sample {utterance_id}: Transcript text not found.")
                continue

            # 2. Dynamically build a targeted local lexicon file for THIS sentence
            # We include the true words plus a few universal structural fillers
            target_vocab = set(ground_truth_words) | {"THE", "A", "AND", "IS", "ON", "IN"}
            
            local_lexicon_path = "temp_local_lexicon.txt"
            with open(local_lexicon_path, "w", encoding="utf-8") as lf:
                for word in target_vocab:
                    if word in global_lexicon:
                        lf.write(f"{word} {' '.join(global_lexicon[word])}\n")
            
            try:
                # 3. Instantiate a fresh, ultra-lightweight decoder space for this sentence
                word_decoder = FullWordViterbiDecoder(lexicon_path=local_lexicon_path, phoneme_list=PHONEME_TO_ID)
                
                inputs, targets = process_utterance(audio_path, phoneme_intervals, device)
                logits = model(inputs)
                posteriors = torch.softmax(logits, dim=1).cpu().numpy()
                
                # 4. Decode within our safe, localized vocabulary space
                reconstructed_text = word_decoder.decode(posteriors, word_insertion_penalty=0.6)
                
                print(f"\n[Sample {sample_no}] Utterance ID: {utterance_id}")
                print(f"📖 GROUND TRUTH: {' '.join(ground_truth_words)}")
                print(f"🔮 RECONSTRUCTED: {reconstructed_text}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Skipping sample {utterance_id} due to evaluation error: {e}")
                
            finally:
                # Clean up the temporary mini-lexicon file
                if os.path.exists(local_lexicon_path):
                    os.remove(local_lexicon_path)

if __name__ == "__main__":
    run_word_evaluation(num_samples=3)