import os
import random
import numpy as np
import copy
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

# ---------------------------------------------------------
# 1. Fixed-Point Quantization Functions
# ---------------------------------------------------------
def quantize_fixed_point(tensor: torch.Tensor, int_bits: int, frac_bits: int, signed: bool = True) -> torch.Tensor:
    """
    Simulates fixed-point quantization Q(int_bits).(frac_bits).
    Total bits = (1 if signed else 0) + int_bits + frac_bits.
    """
    scale = 2.0 ** frac_bits
    if signed:
        total_bits = 1 + int_bits + frac_bits
        qmin = -(2.0 ** (total_bits - 1))
        qmax = (2.0 ** (total_bits - 1)) - 1.0
    else:
        total_bits = int_bits + frac_bits
        qmin = 0.0
        qmax = (2.0 ** total_bits) - 1.0

    # Scale, round, clamp to bit limits, and scale back
    q_tensor = torch.clamp(torch.round(tensor * scale), qmin, qmax)
    return q_tensor / scale


class QuantizedSpeechDNN(nn.Module):
    def __init__(self, base_model: nn.Module, config: dict):
        super().__init__()
        self.config = config
        
        # 1. Deepcopy the base model to preserve its exact structure and weights
        self.model = copy.deepcopy(base_model)
        
        # 2. Quantize weights and biases
        self._quantize_parameters()
        
        # 3. Hook hidden activations if activation quantization is configured
        self._register_activation_hooks()

    def _quantize_parameters(self):
        w_cfg = self.config.get("weights")
        b_cfg = self.config.get("biases")
        
        with torch.no_grad():
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    if w_cfg:
                        module.weight.data = quantize_fixed_point(
                            module.weight.data, w_cfg["int_bits"], w_cfg["frac_bits"], signed=True
                        )
                    if b_cfg and module.bias is not None:
                        module.bias.data = quantize_fixed_point(
                            module.bias.data, b_cfg["int_bits"], b_cfg["frac_bits"], signed=True
                        )

    def _register_activation_hooks(self):
        act_cfg = self.config.get("hidden_act")
        if not act_cfg:
            return

        # Find linear layers to identify intermediate/hidden layers
        linear_layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        hidden_layers = linear_layers[:-1]  # Exclude the final output layer

        def hook_fn(module, input_tensor, output_tensor):
            # Quantize post-ReLU activations (unsigned)
            return quantize_fixed_point(
                output_tensor, act_cfg["int_bits"], act_cfg["frac_bits"], signed=False
            )

        # Hook either ReLU modules or the hidden linear modules
        relu_modules = [m for m in self.model.modules() if isinstance(m, nn.ReLU)]
        if relu_modules:
            for relu in relu_modules[:-1] if len(relu_modules) == len(linear_layers) else relu_modules:
                relu.register_forward_hook(hook_fn)
        else:
            for layer in hidden_layers:
                layer.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_cfg = self.config.get("input")
        if in_cfg:
            x = quantize_fixed_point(x, in_cfg["int_bits"], in_cfg["frac_bits"], signed=True)
            
        return self.model(x)


# ---------------------------------------------------------
# 2. Evaluation Routine
# ---------------------------------------------------------
def evaluate_config(config_name: str, config: dict, base_model, test_samples, global_lexicon, device):
    q_model = QuantizedSpeechDNN(base_model, config).to(device)
    q_model.eval()

    total_frames = 0
    correct_frames = 0
    reconstructions = []

    with torch.no_grad():
        for sample in test_samples:
            inputs, targets = sample["inputs"], sample["targets"]
            outputs = q_model(inputs)

            _, predicted = torch.max(outputs, 1)
            correct_frames += (predicted == targets).sum().item()
            total_frames += targets.size(0)

            # HMM Viterbi Word Decoding
            posteriors = torch.softmax(outputs, dim=1).cpu().numpy()
            local_lex_path = f"temp_lexicon_{random.randint(1000, 9999)}.txt"
            
            with open(local_lex_path, "w", encoding="utf-8") as lf:
                for word in sample["vocab"]:
                    if word in global_lexicon:
                        lf.write(f"{word} {' '.join(global_lexicon[word])}\n")

            try:
                decoder = FullWordViterbiDecoder(lexicon_path=local_lex_path, state_map=STATE_TO_ID)
                recon_text = decoder.decode(posteriors, word_insertion_penalty=1.5, acoustic_scale=0.17)
                reconstructions.append((sample["id"], sample["ref_text"], recon_text))
            finally:
                if os.path.exists(local_lex_path):
                    os.remove(local_lex_path)

    frame_acc = (correct_frames / total_frames) * 100 if total_frames > 0 else 0.0
    return frame_acc, reconstructions


# ---------------------------------------------------------
# 3. Main Experiment Runner
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "sparse_speech_dnn_weights.pth"
    lexicon_file = "expanded_lexicon_3state.txt"

    if not os.path.exists(weights_path) or not os.path.exists(lexicon_file):
        raise FileNotFoundError("Make sure weights and lexicon file exist in current directory.")

    # Load baseline model
    base_model = SpeechDNN(input_dim=880, hidden_dim=1024, output_dim=NUM_CLASSES).to(device)
    base_model.load_state_dict(torch.load(weights_path, map_location=device))
    base_model.eval()

    # Load Lexicon
    global_lexicon = {}
    with open(lexicon_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                global_lexicon[parts[0]] = parts[1:]

    # Preload 5 evaluation utterances to test across all configs
    dataset = LibriSpeechFramesDataset(raw_ds, BASE_LIBRISPEECH_DIR, max_utterances=1000, device=device)
    eval_indices = random.sample(range(len(dataset)), 5)
    test_samples = []

    for idx in eval_indices:
        utt_id = dataset.ids[idx]
        intervals = dataset.phonemes_data[idx]
        spk_id, chap_id, _ = utt_id.split("-")
        audio_path = os.path.join(dataset.base_dir, "train-clean-100", spk_id, chap_id, f"{utt_id}.flac")
        trans_file = os.path.join(dataset.base_dir, "train-clean-100", spk_id, chap_id, f"{spk_id}-{chap_id}.trans.txt")

        ground_truth_words = []
        if os.path.exists(trans_file):
            with open(trans_file, "r") as tf:
                for line in tf:
                    if line.startswith(utt_id):
                        ground_truth_words = line.strip().split()[1:]
                        break
        try:
            inputs, targets = process_utterance(audio_path, intervals, device)
            test_samples.append({
                "id": utt_id,
                "inputs": inputs,
                "targets": targets,
                "vocab": set(ground_truth_words),
                "ref_text": " ".join(ground_truth_words)
            })
        except Exception:
            continue

    # Define Candidate Precision Grids
    # Q(int_bits).(frac_bits)
    candidate_experiments = {
        "Baseline (FP32)": {
            "input": None, "hidden_act": None, "weights": None, "biases": None
        },
        "Exp1_Aggressive_Weights (W:Q1.4, B:Q1.2, Act:Q4.11, In:Q2.13)": { #so far the best one 
            "input": {"int_bits": 2, "frac_bits": 13},       # 16-bit signed
            "hidden_act": {"int_bits": 4, "frac_bits": 11},  # 15-bit unsigned
            "weights": {"int_bits": 1, "frac_bits": 4},      # 5-bit signed
            "biases": {"int_bits": 1, "frac_bits": 3},       # 5-bit signed
        },
        "Exp2_Slightly_Wider_Weights (W:Q1.4, B:Q1.4, Act:Q4.11, In:Q2.13)": {
            "input": {"int_bits": 1, "frac_bits": 13},       # 16-bit signed
            "hidden_act": {"int_bits": 4, "frac_bits": 11},  # 15-bit unsigned
            "weights": {"int_bits": 1, "frac_bits": 4},      # 6-bit signed
            "biases": {"int_bits": 1, "frac_bits": 4},       # 7-bit signed
        },
        "Exp3_High_Dynamic_Range (W:Q2.4, B:Q3.4, Act:Q10.5, In:Q2.13)": { #so far the 2nd best  
            "input": {"int_bits": 2, "frac_bits": 13},       # 16-bit signed
            "hidden_act": {"int_bits": 10, "frac_bits": 5},  # 15-bit unsigned
            "weights": {"int_bits": 2, "frac_bits": 4},      # 7-bit signed
            "biases": {"int_bits": 3, "frac_bits": 4},       # 8-bit signed
        },
        "Exp4_Ultra_Compact (W:Q2.4, B:Q2.4, Act:Q10.5, In:Q2.13)": {
            "input": {"int_bits": 2, "frac_bits": 13},       # 16-bit signed
            "hidden_act": {"int_bits": 10, "frac_bits": 5},  # 15-bit unsigned
            "weights": {"int_bits": 2, "frac_bits": 4},      # 5-bit signed
            "biases": {"int_bits": 2, "frac_bits": 4},       # 5-bit signed
        },
    }

    print("=" * 80)
    print("RUNNING FIXED-POINT PRECISION SWEEP")
    print("=" * 80)

    for name, cfg in candidate_experiments.items():
        print(f"\n--- Testing Configuration: {name} ---")
        acc, recons = evaluate_config(name, cfg, base_model, test_samples, global_lexicon, device)
        print(f"Frame Accuracy: {acc:.2f}%")
        for utt_id, ref, hyp in recons[:2]:  # Show first two utterances
            print(f"[{utt_id}] REF: {ref}")
            print(f"[{utt_id}] HYP: {hyp}")

if __name__ == "__main__":
    main()