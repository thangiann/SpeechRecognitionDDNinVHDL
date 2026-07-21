import os

os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

import torch
import numpy as np

class FullWordViterbiDecoder:
    def __init__(self, lexicon_path, phoneme_list):
        self.phoneme_to_idx = {p: i for i, p in enumerate(phoneme_list)}
        self.word_nodes = {}  # Stores structural data for each word
        self.load_lexicon(lexicon_path)
        self.compile_search_space()

    def load_lexicon(self, path):
        """Parses words and their constituent phonetic state sequences."""
        self.lexicon = {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"Please create a lexicon file at {path}")
            
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    phonemes = parts[1:]
                    # Convert to active model indices, ignoring untracked frames
                    indices = [self.phoneme_to_idx[p] for p in phonemes if p in self.phoneme_to_idx]
                    if indices:
                        self.lexicon[word] = indices

    def compile_search_space(self):
        """Flattens the entire dictionary graph into a continuous index network."""
        self.state_to_word = []
        self.state_phoneme_idx = []
        self.word_boundaries = []  # Tracks (start_idx, end_idx) pointers for each word
        
        curr_idx = 0
        for word, p_indices in self.lexicon.items():
            start = curr_idx
            for p_idx in p_indices:
                self.state_to_word.append(word)
                self.state_phoneme_idx.append(p_idx)
                curr_idx += 1
            end = curr_idx - 1
            self.word_boundaries.append((start, end, word))
            
        self.total_graph_states = len(self.state_to_word)
        self.state_phoneme_idx = np.array(self.state_phoneme_idx)

    def decode(self, dnn_posteriors, word_insertion_penalty=0.001):
        """
        Executes structural word-level token passing across time frames.
        dnn_posteriors: shape (num_frames, 71) matrix from your SpeechDNN
        """
        num_frames = dnn_posteriors.shape[0]
        num_states = self.total_graph_states
        
        # Working log domain structures to avoid floating underflow
        log_B = np.log(dnn_posteriors + 1e-12)
        viterbi = np.full((num_frames, num_states), -np.inf)
        backpointer = np.zeros((num_frames, num_states), dtype=int)
        
        # Variable tracking configurations
        self_loop_cost = np.log(0.85)
        next_step_cost = np.log(0.15)
        log_wip = np.log(word_insertion_penalty)
        
        # Initialization (Frame 0: All word-starting nodes initialized evenly)
        for start, _, _ in self.word_boundaries:
            viterbi[0, start] = log_B[0, self.state_phoneme_idx[start]]

        # Main Forward Token Passing Loop
        for t in range(1, num_frames):
            # Step A: Standard Intra-word Internal Transitions
            for start, end, _ in self.word_boundaries:
                for s in range(start, end + 1):
                    # Path 1: Staying inside the same phonetic node
                    cost_stay = viterbi[t-1, s] + self_loop_cost
                    
                    # Path 2: Transitioning from the previous phonetic node within the word
                    cost_move = -np.inf
                    if s > start:
                        cost_move = viterbi[t-1, s-1] + next_step_cost
                        
                    if cost_stay >= cost_move:
                        viterbi[t, s] = cost_stay + log_B[t, self.state_phoneme_idx[s]]
                        backpointer[t, s] = s
                    else:
                        viterbi[t, s] = cost_move + log_B[t, self.state_phoneme_idx[s]]
                        backpointer[t, s] = s - 1
                        
            # Step B: Inter-word Global Transitions (Connecting word ends back to word starts)
            # Find the best scoring word-ending node from the previous frame
            best_word_end_score = -np.inf
            best_word_end_state = -1
            for _, end, _ in self.word_boundaries:
                if viterbi[t-1, end] > best_word_end_score:
                    best_word_end_score = viterbi[t-1, end]
                    best_word_end_state = end
                    
            # If a valid path exits a word, evaluate updating all starting nodes
            if best_word_end_score > -np.inf:
                cross_word_score = best_word_end_score + log_wip + log_B[t, self.state_phoneme_idx[start]]
                for start, _, _ in self.word_boundaries:
                    # Compare pure path cost vs intra-word cost calculated in Step A
                    if cross_word_score > viterbi[t, start]:
                        viterbi[t, start] = cross_word_score
                        backpointer[t, start] = -best_word_end_state  # Negative marks inter-word leap

        # Backtracking Phase
        best_end_state = np.argmax(viterbi[-1, :])
        state_path = []
        curr_state = best_end_state
        
        for t in range(num_frames - 1, -1, -1):
            state_path.insert(0, curr_state)
            next_ptr = backpointer[t, curr_state]
            if next_ptr < 0:  # Cross-word leap point
                curr_state = int(abs(next_ptr))
            else:
                curr_state = int(next_ptr)
                
        # Turn state tracking IDs into concrete words
        # Updated, noise-robust phrase collapsing logic in hmm.py:
        words_out = []
        last_word = None
        word_frame_count = 0
        MIN_WORD_FRAMES = 3  # Ignores 1-2 frame (10-20ms) transient blips/noise

        for s in state_path:
            word = self.state_to_word[s]
            if word == last_word:
                word_frame_count += 1
            else:
                if last_word is not None and word_frame_count >= MIN_WORD_FRAMES:
                    words_out.append(last_word)
                last_word = word
                word_frame_count = 1

        # Catch the final trailing word
        if last_word is not None and word_frame_count >= MIN_WORD_FRAMES:
            words_out.append(last_word)

        return " ".join(words_out)

# --- EXAMPLE ENGINE INTEGRATION ASSEMBLY HUB ---
if __name__ == "__main__":
    from dnn import PHONEME_MAP  # Pulls your established script map array
    
    # Initialize structural pipeline configuration 
    decoder = FullWordViterbiDecoder(lexicon_path="lexicon.txt", phoneme_list=PHONEME_MAP)
    
    # Simulate a fake DNN output tensor: 120 frames long, 71 classes wide
    fake_logits = torch.randn(120, 71)
    fake_posteriors = torch.softmax(fake_logits, dim=1).numpy()
    
    reconstructed_phrase = decoder.decode(fake_posteriors)
    print(f"Resulting Word Decoding Output: {reconstructed_phrase}")