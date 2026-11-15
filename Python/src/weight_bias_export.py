import os
import torch
import torch.nn as nn
from dnn import NUM_CLASSES, SpeechDNN

# ---------------------------------------------------------------------------
# Fixed-Point Specifications
# Weights: Q1.4 (6 bits signed, range [-2.0, +1.9375], step = 0.0625)
# Biases:  Q1.2 (4 bits signed, range [-2.0, +1.75],   step = 0.25)
# ---------------------------------------------------------------------------
W_FRAC_BITS = 4
W_SCALE = 2.0 ** W_FRAC_BITS
W_QMIN, W_QMAX = -32, 31  # 6-bit signed boundaries

B_FRAC_BITS = 2
B_SCALE = 2.0 ** B_FRAC_BITS
B_QMIN, B_QMAX = -8, 7    # 4-bit signed boundaries (-2^(4-1) to 2^(4-1)-1)


def float_to_q1_4(val: float) -> int:
    scaled = round(val * W_SCALE)
    clamped = max(W_QMIN, min(W_QMAX, scaled))
    return clamped & 0x3F  # 6-bit mask


def float_to_q1_2(val: float) -> int:
    scaled = round(val * B_SCALE)
    clamped = max(B_QMIN, min(B_QMAX, scaled))
    return clamped & 0x0F  # 4-bit mask


def export_weights_and_biases(
    checkpoint_path="sparse_speech_dnn_weights.pth",
    output_dir="ram_init_16x6"
):
    device = torch.device("cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = SpeechDNN(input_dim=880, hidden_dim=1024, output_dim=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    # 16 separate weight streams
    weight_rams = [[] for _ in range(16)]
    # 1 single bias stream
    bias_ram = []

    current_weight_addr = 0
    current_bias_addr = 0

    print("=" * 80)
    print(" EXPORTING QUANTIZED WEIGHTS (16x 6-bit) & BIASES (1x 4-bit)")
    print("=" * 80)

    for l_idx, layer in enumerate(linear_layers):
        out_f, in_f = layer.weight.shape

        # --- 1. PROCESS WEIGHTS (Non-zero only, aligned to 16) ---
        weights = layer.weight.detach().numpy().flatten()
        non_zero_weights = list(weights[weights != 0.0])

        remainder = len(non_zero_weights) % 16
        if remainder != 0:
            non_zero_weights.extend([0.0] * (16 - remainder))

        words = len(non_zero_weights) // 16
        for w_idx in range(words):
            chunk = non_zero_weights[w_idx * 16 : (w_idx + 1) * 16]
            for mac_id in range(16):
                weight_rams[mac_id].append(float_to_q1_4(chunk[mac_id]))
            current_weight_addr += 1

        # --- 2. PROCESS BIASES ---
        bias_start = current_bias_addr
        if layer.bias is not None:
            biases = list(layer.bias.detach().numpy())
            
            # If this is the output layer (213 classes), pad to 224 (multiple of 16)
            if out_f == 213:
                biases.extend([0.0] * 11)  # 213 + 11 = 224

            for b in biases:
                bias_ram.append(float_to_q1_2(b))
                current_bias_addr += 1
            
            bias_end = current_bias_addr - 1
            print(f"Layer {l_idx} ({in_f} -> {out_f}): Biases mapped to Addr {bias_start}..{bias_end} ({len(biases)} values)")

    # --- WRITE 16 WEIGHT RAM FILES ---
    for mac_id in range(16):
        path = os.path.join(output_dir, f"ram_weights_mac{mac_id:02d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("BEGIN\n")
            for addr, val in enumerate(weight_rams[mac_id]):
                f.write(f"    {addr} : {val:02X};\n")
            f.write("END;\n")

    # --- WRITE 1 BIAS RAM FILE ---
    bias_path = os.path.join(output_dir, "bias_ram.txt")
    with open(bias_path, "w", encoding="utf-8") as f:
        f.write("BEGIN\n")
        for addr, val in enumerate(bias_ram):
            # 4 bits formatted as 1-character hex (0 to F)
            f.write(f"    {addr} : {val:01X};\n")
        f.write("END;\n")

    print("-" * 80)
    print(f"Successfully generated:")
    print(f"  - 16 Weight RAM files: 'ram_weights_mac00.txt' to 'mac15.txt' (Depth: {current_weight_addr})")
    print(f"  - 1 Bias RAM file:     'bias_ram.txt' (Depth: {current_bias_addr})")
    print(f"Saved in folder: '{output_dir}/'\n")


if __name__ == "__main__":
    export_weights_and_biases()