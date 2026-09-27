import os
import numpy as np
import torch

weights_path = "sparse_speech_dnn_weights.pth"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Cannot find {weights_path}")

state_dict = torch.load(weights_path, map_location="cpu")
block_size = 64

print("=" * 80)
print(f"FULL MODEL WEIGHT INSPECTION (ALL LAYERS): {weights_path}")
print("=" * 80)

for param_name, tensor in state_dict.items():
    if "weight" in param_name and tensor.ndim == 2:
        out_f, in_f = tensor.shape
        W = tensor.numpy()

        total_weights = W.size
        zero_weights = np.sum(W == 0.0)
        sparsity_pct = (zero_weights / total_weights) * 100.0

        w_min = float(W.min())
        w_max = float(W.max())
        w_mean = float(W.mean())
        w_std = float(W.std())

        print(f"\nLayer: {param_name} | Matrix Shape: {out_f}x{in_f}")
        print(f"Total Weights: {total_weights:,}")
        print(
            f"Zero Weights: {zero_weights:,} | Sparsity: {sparsity_pct:.2f}%"
        )
        print(
            f"Stats: Min={w_min:.4f}, Max={w_max:.4f}, Mean={w_mean:.4f}, Std={w_std:.4f}"
        )

        # Check if compatible with 64x64 block structure
        if out_f % block_size == 0 and in_f % block_size == 0:
            b_r = out_f // block_size
            b_c = in_f // block_size
            total_blocks = b_r * b_c

            active_blocks = 0
            for r in range(b_r):
                for c in range(b_c):
                    tile = W[
                        r * block_size : (r + 1) * block_size,
                        c * block_size : (c + 1) * block_size,
                    ]
                    if not np.all(tile == 0.0):
                        active_blocks += 1

            print(
                f"-> 64x64 Block Map: {active_blocks}/{total_blocks} Active Blocks "
                f"({(1.0 - active_blocks/total_blocks)*100:.1f}% Block Sparsity)"
            )
        else:
            print(
                "-> Structure: Dense (Irregular shape not divisible by 64x64)"
            )

print("\n" + "=" * 80)