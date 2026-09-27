import os
import urllib.request

# Import from your main dnn script
from dnn import STATE_TO_ID, UNIQUE_PHONEMES


def build_3state_lexicon(output_path="expanded_lexicon_3state.txt"):
    valid_phonemes = set(UNIQUE_PHONEMES)
    cmudict_url = (
        "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    )
    local_cmu_path = "cmudict.dict"

    if not os.path.exists(local_cmu_path):
        print("Downloading CMU Pronouncing Dictionary...")
        urllib.request.urlretrieve(cmudict_url, local_cmu_path)
        print("Download complete.")

    lexicon_entries = {}
    skipped_count = 0
    mismatched_phones = set()

    print("Processing dictionary rules for 3-state expansion...")
    with open(local_cmu_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue

            parts = line.split()
            word = parts[0].split("(")[0].upper()
            raw_phonemes = parts[1:]

            cleaned_phonemes = []
            valid_sequence = True

            for p in raw_phonemes:
                # 1. First try direct match (e.g. AH0, AH1)
                if p in valid_phonemes:
                    cleaned_phonemes.append(p)
                else:
                    # 2. Try stripping stress numbers (e.g. AH1 -> AH)
                    p_no_num = "".join([c for c in p if not c.isdigit()])
                    if p_no_num in valid_phonemes:
                        cleaned_phonemes.append(p_no_num)
                    else:
                        valid_sequence = False
                        mismatched_phones.add(p)
                        break

            if valid_sequence and word.isalpha():
                # Expand into 3 sub-states: P_0 P_1 P_2
                three_state_sequence = []
                for p in cleaned_phonemes:
                    three_state_sequence.extend(
                        [f"{p}_0", f"{p}_1", f"{p}_2"]
                    )

                lexicon_entries[word] = " ".join(three_state_sequence)
            else:
                skipped_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for word, p_seq in sorted(lexicon_entries.items()):
            f.write(f"{word} {p_seq}\n")

    print(f"\n✅ 3-State Expanded Lexicon compiled successfully!")
    print(f"Total vocabulary words loaded: {len(lexicon_entries)}")
    print(f"Skipped words: {skipped_count}")
    if mismatched_phones:
        print(
            f"Sample mismatched phonemes found: {list(mismatched_phones)[:10]}"
        )
    print(f"Saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    build_3state_lexicon()