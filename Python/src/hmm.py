import os

os.environ["HF_HOME"] = r"D:\HuggingFaceCache"

import torch
import numpy as np

class FullWordViterbiDecoder:
    def __init__(self, lexicon_path, phoneme_list):
        if isinstance(phoneme_list, dict):
            self.phoneme_to_idx = phoneme_list
        else:
            self.phoneme_to_idx = {p: i for i, p in enumerate(phoneme_list)}
            
        self.word_nodes = {}
        self.load_lexicon(lexicon_path)
        self.compile_search_space()

    def load_lexicon(self, path):
        """Parses words and their constituent phonetic state sequences."""
        self.lexicon = {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"Please create a lexicon file at {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    phonemes = parts[1:]
                    indices = [self.phoneme_to_idx[p] for p in phonemes if p in self.phoneme_to_idx]
                    if indices:
                        self.lexicon[word] = indices

    def compile_search_space(self):
        """Flattens the entire dictionary graph into a continuous index network."""
        self.state_to_word = []
        self.state_phoneme_idx = []
        self.word_boundaries = []
        self.word_phoneme_counts = {}
        
        curr_idx = 0
        for word, p_indices in self.lexicon.items():
            start = curr_idx
            for p_idx in p_indices:
                self.state_to_word.append(word)
                self.state_phoneme_idx.append(p_idx)
                curr_idx += 1
            end = curr_idx - 1
            self.word_boundaries.append((start, end, word))
            self.word_phoneme_counts[word] = len(p_indices)
            
        self.total_graph_states = len(self.state_to_word)
        self.state_phoneme_idx = np.array(self.state_phoneme_idx)

    def decode(self, dnn_posteriors, word_insertion_penalty=5e-4, acoustic_scale=0.15):
        """
        Executes standard structural word-level token passing with post-emission filtering.
        """
        num_frames = dnn_posteriors.shape[0]
        num_states = self.total_graph_states
        
        # Scale log-posteriors
        log_B = np.log(dnn_posteriors + 1e-12) * acoustic_scale
        
        viterbi = np.full((num_frames, num_states), -np.inf)
        backpointer = np.zeros((num_frames, num_states), dtype=int)
        
        # Standard transition probabilities
        self_loop_cost = np.log(0.88)
        next_step_cost = np.log(0.12)
        log_wip = np.log(word_insertion_penalty)
        
        # Initialize Frame 0
        for start, _, _ in self.word_boundaries:
            viterbi[0, start] = log_B[0, self.state_phoneme_idx[start]]

        # Standard Forward Token Passing Loop
        for t in range(1, num_frames):
            # Step A: Intra-word Transitions
            for start, end, _ in self.word_boundaries:
                for s in range(start, end + 1):
                    cost_stay = viterbi[t-1, s] + self_loop_cost
                    
                    cost_move = -np.inf
                    if s > start:
                        cost_move = viterbi[t-1, s-1] + next_step_cost
                        
                    if cost_stay >= cost_move:
                        viterbi[t, s] = cost_stay + log_B[t, self.state_phoneme_idx[s]]
                        backpointer[t, s] = s
                    else:
                        viterbi[t, s] = cost_move + log_B[t, self.state_phoneme_idx[s]]
                        backpointer[t, s] = s - 1
                        
            # Step B: Inter-word Global Transitions
            best_word_end_score = -np.inf
            best_word_end_state = -1
            for _, end, _ in self.word_boundaries:
                if viterbi[t-1, end] > best_word_end_score:
                    best_word_end_score = viterbi[t-1, end]
                    best_word_end_state = end
                    
            if best_word_end_score > -np.inf:
                for start, _, _ in self.word_boundaries:
                    cross_word_score = best_word_end_score + log_wip + log_B[t, self.state_phoneme_idx[start]]
                    if cross_word_score > viterbi[t, start]:
                        viterbi[t, start] = cross_word_score
                        backpointer[t, start] = -best_word_end_state

        # Backtracking Phase
        best_end_state = np.argmax(viterbi[-1, :])
        state_path = []
        curr_state = best_end_state
        
        for t in range(num_frames - 1, -1, -1):
            state_path.insert(0, curr_state)
            next_ptr = backpointer[t, curr_state]
            if next_ptr < 0:
                curr_state = int(abs(next_ptr))
            else:
                curr_state = int(next_ptr)
                
        # Word emission
        raw_words = []
        last_word = None
        word_frame_count = 0

        for s in state_path:
            word = self.state_to_word[s]
            if word == last_word:
                word_frame_count += 1
            else:
                if last_word is not None:
                    # Require short words (<=2 phonemes) to hold >= 4 frames (40ms)
                    min_frames = 4 if self.word_phoneme_counts.get(last_word, 3) <= 2 else 3
                    if word_frame_count >= min_frames:
                        raw_words.append(last_word)
                last_word = word
                word_frame_count = 1

        if last_word is not None:
            min_frames = 4 if self.word_phoneme_counts.get(last_word, 3) <= 2 else 3
            if word_frame_count >= min_frames:
                raw_words.append(last_word)

        # Cleanup Phase: Remove edge noise & repeated short auxiliary words
        cleaned_words = []
        for w in raw_words:
            # Skip consecutive repetitions of short 1-2 phoneme words (e.g. 'HAD HAD')
            is_short = self.word_phoneme_counts.get(w, 3) <= 2
            if is_short and cleaned_words and cleaned_words[-1] == w:
                continue
            cleaned_words.append(w)

        # Trim isolated short filler words at the extreme start/end of the sentence
        while cleaned_words and self.word_phoneme_counts.get(cleaned_words[0], 3) <= 2:
            cleaned_words.pop(0)
        while cleaned_words and self.word_phoneme_counts.get(cleaned_words[-1], 3) <= 2:
            cleaned_words.pop()

        return " ".join(cleaned_words)