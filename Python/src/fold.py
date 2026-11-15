import os
import torch
import torch.nn as nn
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "sparse_speech_dnn_weights.pth"
LEXICON_PATH = "expanded_lexicon_3state.txt"


def build_folded_linear_model(base_model, dataset, num_calib=30):
    """
    Measures empirical mean and variance for every LayerNorm across calibration
    utterances, folds the statistics into the preceding Linear layer, and returns
    a pure Linear + ReLU model with zero normalization layers.
    """
    base_model.eval()
    modules = list(base_model.network.children())
    
    # Locate indices of Linear and LayerNorm modules
    ln_indices = [i for i, m in enumerate(modules) if isinstance(m, nn.LayerNorm)]
    ln_stats = {idx: {"means": [], "vars": []} for idx in ln_indices}

    # Hook inputs entering each LayerNorm to record runtime distributions
    hooks = []
    def make_hook(idx):
        def hook_fn(module, input_tensor, output_tensor):
            x = input_tensor[0].detach()  # Shape: [num_frames, 1024]
            mu = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            ln_stats[idx]["means"].append(mu.mean().item())
            ln_stats[idx]["vars"].append(var.mean().item())
        return hook_fn

    for idx in ln_indices:
        hooks.append(modules[idx].register_forward_hook(make_hook(idx)))

    print(f"[*] Calibrating empirical LayerNorm statistics over {num_calib} utterances...")
    valid_calib = 0
    with torch.no_grad():
        for i in range(min(num_calib * 2, len(dataset))):
            if valid_calib >= num_calib:
                break
            utt_id = dataset.ids[i]
            intervals = dataset.phonemes_data[i]
            spk, chap, _ = utt_id.split("-")
            audio_path = os.path.join(dataset.base_dir, "train-clean-100", spk, chap, f"{utt_id}.flac")
            try:
                inputs, _ = process_utterance(audio_path, intervals, device)
                _ = base_model(inputs)
                valid_calib += 1
            except Exception:
                continue

    for h in hooks:
        h.remove()

    print(f"[*] Calibration complete ({valid_calib} utterances processed).")

    # Construct the folded sequence (Linear + ReLU only)
    folded_layers = []
    current_linear = None

    for i, mod in enumerate(modules):
        if isinstance(mod, nn.Linear):
            current_linear = nn.Linear(
                mod.in_features, mod.out_features, bias=True, device=device
            )
            current_linear.weight.data.copy_(mod.weight.data)
            current_linear.bias.data.copy_(mod.bias.data)
            folded_layers.append(current_linear)
        elif isinstance(mod, nn.LayerNorm):
            mean_mu = sum(ln_stats[i]["means"]) / len(ln_stats[i]["means"])
            mean_sigma = (sum(ln_stats[i]["vars"]) / len(ln_stats[i]["vars"]) + mod.eps) ** 0.5
            
            gamma = mod.weight.data.to(device)
            beta = mod.bias.data.to(device)

            scale_factor = gamma / mean_sigma
            current_linear.weight.data = current_linear.weight.data * scale_factor.unsqueeze(1)
            current_linear.bias.data = (current_linear.bias.data - mean_mu) * scale_factor + beta
        elif isinstance(mod, nn.ReLU):
            folded_layers.append(nn.ReLU())
        elif isinstance(mod, nn.Dropout):
            continue

    folded_model = nn.Sequential(*folded_layers).to(device)
    folded_model.eval()
    return folded_model


def run_evaluation(num_test_utts=5):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint '{CHECKPOINT_PATH}' not found.")
    if not os.path.exists(LEXICON_PATH):
        raise FileNotFoundError(f"Lexicon '{LEXICON_PATH}' not found.")

    print(f"[*] Using compute device: {device}")
    
    # 1. Load calibration & test dataset
    dataset = LibriSpeechFramesDataset(raw_ds, BASE_LIBRISPEECH_DIR, max_utterances=1000, device=device)
    
    # 2. Instantiate and load baseline model
    base_model = SpeechDNN(input_dim=880, hidden_dim=1024, output_dim=NUM_CLASSES).to(device)
    base_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    base_model.eval()

    # 3. Create folded model
    folded_model = build_folded_linear_model(base_model, dataset, num_calib=30)

    # 4. Load vocabulary lexicon
    global_lexicon = {}
    with open(LEXICON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                global_lexicon[parts[0]] = parts[1:]

    # 5. Comparative evaluation
    print("\n" + "=" * 80)
    print(" BASELINE (DYNAMIC LAYERNORM) vs. FOLDED (PURE LINEAR MVM)")
    print("=" * 80)

    base_correct, fold_correct, total_frames = 0, 0, 0
    tested_count = 0

    # Start after calibration set to evaluate on unseen utterances
    start_eval_idx = 40

    with torch.no_grad():
        for idx in range(start_eval_idx, len(dataset)):
            if tested_count >= num_test_utts:
                break

            utt_id = dataset.ids[idx]
            intervals = dataset.phonemes_data[idx]
            spk, chap, _ = utt_id.split("-")
            audio_path = os.path.join(dataset.base_dir, "train-clean-100", spk, chap, f"{utt_id}.flac")
            trans_file = os.path.join(dataset.base_dir, "train-clean-100", spk, chap, f"{spk}-{chap}.trans.txt")

            ref_words = []
            if os.path.exists(trans_file):
                with open(trans_file, "r") as tf:
                    for line in tf:
                        if line.startswith(utt_id):
                            ref_words = line.strip().split()[1:]
                            break

            try:
                inputs, targets = process_utterance(audio_path, intervals, device)
            except Exception:
                continue

            # Forward passes
            out_base = base_model(inputs)
            out_fold = folded_model(inputs)

            _, pred_base = torch.max(out_base, 1)
            _, pred_fold = torch.max(out_fold, 1)

            frames = targets.size(0)
            total_frames += frames
            base_correct += (pred_base == targets).sum().item()
            fold_correct += (pred_fold == targets).sum().item()

            inst_base_acc = ((pred_base == targets).sum().item() / frames) * 100
            inst_fold_acc = ((pred_fold == targets).sum().item() / frames) * 100

            # Decode sentences with Viterbi
            temp_lex = f"temp_eval_lex_{tested_count}.txt"
            with open(temp_lex, "w", encoding="utf-8") as lf:
                for w in set(ref_words):
                    if w in global_lexicon:
                        lf.write(f"{w} {' '.join(global_lexicon[w])}\n")

            decoder = FullWordViterbiDecoder(lexicon_path=temp_lex, state_map=STATE_TO_ID)
            post_base = torch.softmax(out_base, dim=1).cpu().numpy()
            post_fold = torch.softmax(out_fold, dim=1).cpu().numpy()

            hyp_base = decoder.decode(post_base, word_insertion_penalty=1.5, acoustic_scale=0.17)
            hyp_fold = decoder.decode(post_fold, word_insertion_penalty=1.5, acoustic_scale=0.17)

            if os.path.exists(temp_lex):
                os.remove(temp_lex)

            print(f"\n[Test Sample {tested_count + 1}: {utt_id}]")
            print(f"  Frame Acc:  Baseline = {inst_base_acc:.2f}%  |  Folded = {inst_fold_acc:.2f}%")
            print(f"  REFERENCE:  {' '.join(ref_words)}")
            print(f"  BASELINE:   {hyp_base}")
            print(f"  FOLDED:     {hyp_fold}")

            tested_count += 1

    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK")
    print("=" * 80)
    base_total_acc = (base_correct / total_frames) * 100 if total_frames > 0 else 0
    fold_total_acc = (fold_correct / total_frames) * 100 if total_frames > 0 else 0
    print(f"Evaluated Frames:       {total_frames}")
    print(f"Baseline Overall Acc:   {base_total_acc:.2f}%")
    print(f"Folded Overall Acc:     {fold_total_acc:.2f}%")
    print(f"Delta:                  {fold_total_acc - base_total_acc:+.2f}%\n")


if __name__ == "__main__":
    run_evaluation(num_test_utts=5)