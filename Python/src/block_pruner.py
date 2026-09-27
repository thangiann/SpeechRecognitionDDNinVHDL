import torch


class BlockPruner:

    def __init__(self, model, block_size=64, keep_ratio=0.25):
        self.model = model
        self.block_size = block_size
        self.keep_ratio = keep_ratio
        self.masks = {}
        self.init_masks()

    def init_masks(self):
        """Generates 64x64 binary block masks for all compatible Linear layers."""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    W = module.weight
                    out_f, in_f = W.shape

                    # Check if layer dimensions are divisible by block_size
                    if (
                        out_f % self.block_size == 0
                        and in_f % self.block_size == 0
                    ):
                        b_r = out_f // self.block_size
                        b_c = in_f // self.block_size

                        # Reshape to [b_r, block_size, b_c, block_size]
                        blocks = W.view(
                            b_r, self.block_size, b_c, self.block_size
                        )
                        # Compute Frobenius norm per block: shape [b_r, b_c]
                        block_norms = torch.norm(blocks, p=2, dim=(1, 3))

                        total_blocks = b_r * b_c
                        k = int(total_blocks * self.keep_ratio)

                        # Find threshold for top-k blocks
                        flat_norms = block_norms.flatten()
                        threshold = torch.topk(flat_norms, k=k).values[-1]

                        # Block-level binary mask
                        block_mask = (block_norms >= threshold).float()

                        # Expand back to original weight shape [out_f, in_f]
                        expanded_mask = (
                            block_mask.unsqueeze(1)
                            .unsqueeze(3)
                            .repeat(1, self.block_size, 1, self.block_size)
                            .view(out_f, in_f)
                        )

                        self.masks[name] = expanded_mask.to(W.device)
                        print(
                            f"Layer '{name}' [{out_f}x{in_f}]: Kept {k}/{total_blocks} blocks "
                            f"({k/total_blocks*100:.1f}%)"
                        )

    def apply_mask(self):
        """Zeroes out pruned weights according to the precomputed masks."""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.masks:
                    module.weight.data.mul_(self.masks[name])

    def zero_masked_gradients(self):
        """Prevents zeroed blocks from receiving gradient updates."""
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if name in self.masks and module.weight.grad is not None:
                    module.weight.grad.data.mul_(self.masks[name])