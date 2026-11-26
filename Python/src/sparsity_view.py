import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuration
CHECKPOINT_PATH = "sparse_speech_dnn_weights.pth"
BLOCK_SIZE = 64

# Select the layer you want to visualize (e.g., Hidden Layer 1)
# You may need to adjust this key (e.g., 'network.5.weight') based on your state_dict
LAYER_NAME = "network.20.weight" 

def visualize_blocks():
    # Load weights
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if LAYER_NAME not in state_dict:
        print(f"Keys in model: {list(state_dict.keys())}")
        raise KeyError(f"Layer '{LAYER_NAME}' not found in checkpoint.")
        
    weight_tensor = state_dict[LAYER_NAME]
    out_features, in_features = weight_tensor.shape
    
    # Calculate how many 64x64 blocks exist in each dimension
    blocks_rows = out_features // BLOCK_SIZE  # 'i' dimension (16 groups of 64 neurons)
    blocks_cols = in_features // BLOCK_SIZE   # 'j' dimension (16 groups of 64 inputs)
    
    block_map = np.zeros((blocks_rows, blocks_cols), dtype=int)
    
    active_i = []
    active_j = []
    
    # 2. Scan the matrix block by block
    for i in range(blocks_rows):
        for j in range(blocks_cols):
            # Slice out the specific 64x64 block
            block = weight_tensor[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE, 
                                  j*BLOCK_SIZE : (j+1)*BLOCK_SIZE]
            
            # If the block has non-zero elements, it survived pruning
            if torch.count_nonzero(block).item() > 0:
                block_map[i, j] = 1
                active_i.append(i)
                active_j.append(j)
                
    # 3. Print the hardware metadata arrays
    total_blocks = blocks_rows * blocks_cols
    print(f"Layer: {LAYER_NAME} ({out_features}x{in_features})")
    print(f"Total Active Blocks: {len(active_i)} / {total_blocks}")
    print(f"Actual Sparsity: {100 * (1.0 - len(active_i)/total_blocks):.2f}%\n")
    
    print("active_i = [")
    print("  " + ", ".join(map(str, active_i)))
    print("]")
    
    print("\nactive_j = [")
    print("  " + ", ".join(map(str, active_j)))
    print("]")
    
    # 4. Plot the Block Map
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(block_map, cmap="Blues", vmin=0, vmax=1)
    
    # Draw gridlines to clearly separate the blocks
    ax.set_xticks(np.arange(-0.5, blocks_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, blocks_rows, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", size=0)
    
    ax.set_xticks(np.arange(blocks_cols))
    ax.set_yticks(np.arange(blocks_rows))
    ax.set_xlabel("Input Feature Block Group (j)")
    ax.set_ylabel("Output Neuron Block Group (i)")
    ax.set_title(f"64x64 Block Sparsity Map\n(Dark = Active, White = Pruned)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_blocks()