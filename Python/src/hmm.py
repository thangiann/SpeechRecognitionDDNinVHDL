import os
import numpy as np
import torch
from dnn import STATE_TO_ID


class FullWordViterbiDecoder:

    def __init__(
        self, lexicon_path, state_map=STATE_TO_ID, class_priors=None
    ):
        self.state_to_idx = state_map
        self.class_priors = (
            np.log(class_priors + 1e-8) if class_priors is not None else 0.0
        )
        self.word_nodes = {}
        self.load_lexicon(lexicon_path)
        self.compile_search_space()

    def load_lexicon(self, path):
        self.lexicon = {}
        if not os.path.exists(path):
            raise FileNotFoundError(f"Lexicon not found at {path}")

        # Ensure explicit silence entry exists
        if "SIL" in self.state_to_idx:
            self.lexicon["<SIL>"] = [self.state_to_idx["SIL"]]

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0]
                    state_tokens = parts[1:]
                    indices = [
                        self.state_to_idx[s]
                        for s in state_tokens
                        if s in self.state_to_idx
                    ]
                    if indices:
                        self.lexicon[word] = indices

    def compile_search_space(self):
        self.state_to_word = []
        self.state_target_idx = []
        self.word_start_mask = []
        self.word_end_indices = []
        self.word_start_indices = []

        curr_idx = 0
        for word, s_indices in self.lexicon.items():
            start = curr_idx
            for i, s_idx in enumerate(s_indices):
                self.state_to_word.append(word)
                self.state_target_idx.append(s_idx)
                self.word_start_mask.append(i == 0)
                curr_idx += 1
            end = curr_idx - 1
            self.word_start_indices.append(start)
            self.word_end_indices.append(end)

        self.total_graph_states = len(self.state_to_word)
        self.state_target_idx = np.array(self.state_target_idx, dtype=np.int32)
        self.word_start_mask = np.array(self.word_start_mask, dtype=bool)
        self.word_end_indices = np.array(self.word_end_indices, dtype=np.int32)
        self.word_start_indices = np.array(
            self.word_start_indices, dtype=np.int32
        )

        print(
            f"Decoder Graph Initialized with {self.total_graph_states} active states."
        )

    def decode(
        self,
        dnn_posteriors,
        word_insertion_penalty=0.8,
        acoustic_scale=0.14,
    ):
        num_frames = dnn_posteriors.shape[0]
        num_states = self.total_graph_states

        # Bayes rule normalization: log P(x|q) = log P(q|x) - log P(q)
        log_posteriors = np.log(dnn_posteriors + 1e-12) - self.class_priors
        log_B = log_posteriors * acoustic_scale

        viterbi = np.full((num_frames, num_states), -np.inf)
        backpointer = np.zeros((num_frames, num_states), dtype=np.int32)

        self_loop_cost = np.log(0.50)
        next_step_cost = np.log(0.50)
        log_wip = np.log(word_insertion_penalty)

        # Initialize Frame 0
        starts = self.word_start_indices
        viterbi[0, starts] = log_B[0, self.state_target_idx[starts]]

        # Forward Token Passing Loop
        for t in range(1, num_frames):
            prev_v = viterbi[t - 1]

            # Vectorized intra-word transitions
            cost_stay = prev_v + self_loop_cost
            cost_move = np.full(num_states, -np.inf)
            cost_move[1:] = prev_v[:-1] + next_step_cost
            cost_move[self.word_start_mask] = (
                -np.inf
            )  # Block invalid inter-word bleed

            stay_wins = cost_stay >= cost_move
            best_intra = np.where(stay_wins, cost_stay, cost_move)

            viterbi[t] = best_intra + log_B[t, self.state_target_idx]
            backpointer[t] = np.where(
                stay_wins,
                np.arange(num_states, dtype=np.int32),
                np.arange(num_states, dtype=np.int32) - 1,
            )

            # Inter-word Global Transitions
            best_end_idx = np.argmax(prev_v[self.word_end_indices])
            best_word_end_score = prev_v[self.word_end_indices[best_end_idx]]
            best_word_end_state = self.word_end_indices[best_end_idx]

            if best_word_end_score > -np.inf:
                cross_score = (
                    best_word_end_score
                    + log_wip
                    + log_B[t, self.state_target_idx[starts]]
                )
                better_cross = cross_score > viterbi[t, starts]

                # Update starting states that benefit from word transition
                target_starts = starts[better_cross]
                viterbi[t, target_starts] = cross_score[better_cross]

                # Offset encoding to prevent sign bug on index 0
                backpointer[t, target_starts] = -(best_word_end_state + 1)

        # Backtracking Phase
        best_end_state = np.argmax(viterbi[-1])
        curr_state = int(best_end_state)
        state_path = []

        for t in range(num_frames - 1, -1, -1):
            state_path.insert(0, curr_state)
            next_ptr = backpointer[t, curr_state]
            if next_ptr < 0:
                curr_state = int(abs(next_ptr) - 1)
            else:
                curr_state = int(next_ptr)

        # Reconstruct words
        recognized_words = []
        last_word = None
        for s in state_path:
            word = self.state_to_word[s]
            if word != last_word:
                if word and word != "<SIL>":
                    recognized_words.append(word)
                last_word = word

        return " ".join(recognized_words)