import os
from dnn import UNIQUE_PHONEMES

def verify_lexicon_file(lexicon_path="expanded_lexicon.txt"):
    # Convert map to a set for instant O(1) lookups
    valid_phonemes = set(UNIQUE_PHONEMES)
    
    if not os.path.exists(lexicon_path):
        print(f"❌ Error: Could not find {lexicon_path}")
        return

    missing_tokens = set()
    total_words = 0
    corrupted_lines = 0

    print("--- Starting Lexicon Phoneme Validation ---")
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines or comments
                
            parts = line.split()
            word = parts[0]
            phonemes = parts[1:]
            total_words += 1

            if not phonemes:
                print(f"⚠️ Warning [Line {line_num}]: Word '{word}' has no phonemes defined.")
                corrupted_lines += 1
                continue

            for p in phonemes:
                if p not in valid_phonemes:
                    missing_tokens.add(p)
                    print(f"❌ Mismatch [Line {line_num}]: Word '{word}' uses unknown phoneme token '{p}'")

    print("\n--- Summary Statistics ---")
    print(f"Total words parsed: {total_words}")
    
    if missing_tokens:
        print(f"🚨 FAILED: Found {len(missing_tokens)} unique phoneme tokens not recognized by dnn.py!")
        print(f"Invalid tokens to fix or add to PHONEME_MAP: {list(missing_tokens)}")
    else:
        print("✅ SUCCESS: All phonemes in your lexicon match your DNN vocabulary perfectly!")

if __name__ == "__main__":
    verify_lexicon_file()