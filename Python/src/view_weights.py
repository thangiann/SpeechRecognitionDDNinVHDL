import numpy as np
import torch

weights_path = "sparse_speech_dnn_weights.pth"
state_dict = torch.load(weights_path, map_location="cpu")

# Choose layer to dump
layer_name = "network.4.weight"
W = state_dict[layer_name].numpy()

output_filename = "layer_weights_full_dump.txt"
print(f"Dumping {W.shape[0]}x{W.shape[1]} ({W.size:,} weights) to '{output_filename}'...")

# Write complete floating-point matrix formatted with 4 decimal places
np.savetxt(output_filename, W, fmt="%.4f", delimiter="\t")

print("Done! You can now open 'layer_weights_full_dump.txt' in your editor.")