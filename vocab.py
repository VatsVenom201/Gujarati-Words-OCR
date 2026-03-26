import os
from config import TRAIN_GT

def build_vocab(train_gt_path):
    print(f"Building vocabulary from {train_gt_path}...")
    
    unique_chars = set()
    
    with open(train_gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 2:
                label = parts[1]
                # list(label) naturally splits the string into characters/Unicode symbols
                for char in list(label):
                    unique_chars.add(char)
                    
    # Sort for deterministic vocabulary
    sorted_chars = sorted(list(unique_chars))
    
    # CTC Requirement: explicitly reserve index 0 for <blank>
    char_to_idx = {'<blank>': 0}
    idx_to_char = {0: '<blank>'}
    
    for idx, char in enumerate(sorted_chars, start=1):
        char_to_idx[char] = idx
        idx_to_char[idx] = char
        
    print(f"Vocabulary built! Total characters: {len(char_to_idx)} (including <blank>)")
    
    return char_to_idx, idx_to_char

if __name__ == "__main__":
    c2i, i2c = build_vocab(TRAIN_GT)
    print("Example Mappings:")
    for i in range(5):
        if i in i2c:
            print(f"{i} -> {i2c[i]}")
