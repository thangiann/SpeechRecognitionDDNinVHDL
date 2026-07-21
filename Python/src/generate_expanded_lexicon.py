import urllib.request
import os
from dnn import PHONEME_TO_ID

def build_filtered_lexicon(output_path="expanded_lexicon.txt"):
    valid_phonemes = set(PHONEME_TO_ID)
    cmudict_url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    local_cmu_path = "cmudict.dict"
    
    # 1. Download the reference CMU Dictionary if not locally present
    if not os.path.exists(local_cmu_path):
        print("Downloading CMU Pronouncing Dictionary...")
        urllib.request.urlretrieve(cmudict_url, local_cmu_path)
        print("Download complete.")
        
    lexicon_entries = {}
    mismatched_tokens = set()
    
    print("Processing dictionary rules...")
    with open(local_cmu_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"): # Skip headers
                continue
                
            parts = line.split()
            word = parts[0].split('(')[0].upper() # Strip alternative pronunciation numbers e.g. CAT(1)
            phonemes = parts[1:]
            
            # Verify every single phoneme matches the exact stress-marked cases inside dnn.py
            valid_sequence = True
            cleaned_phonemes = []
            
            for p in phonemes:
                # If your unique_phonemes uses stress markers like AH0, AH1, etc.,
                # CMUdict matches natively. If there's a minor variant, catch it here.
                if p in valid_phonemes:
                    cleaned_phonemes.append(p)
                else:
                    valid_sequence = False
                    mismatched_tokens.add(p)
                    break
            
            if valid_sequence and word.isalpha():
                lexicon_entries[word] = " ".join(cleaned_phonemes)

    # 2. Write the verified system lexicon out
    with open(output_path, 'w', encoding='utf-8') as f:
        for word, p_seq in sorted(lexicon_entries.items()):
            f.write(f"{word} {p_seq}\n")
            
    print(f"\n✅ Expanded Lexicon compiled successfully!")
    print(f"Total vocabulary words loaded into space: {len(lexicon_entries)}")
    print(f"Saved directly to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    build_filtered_lexicon()