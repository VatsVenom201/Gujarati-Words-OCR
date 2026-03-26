import torch

def greedy_decode(preds, idx_to_char, blank=0):
    """
    Decodes the model predictions according to CTC logic.
    preds: tensor of shape (B, T, V) or (B, T)
    idx_to_char: dictionary mapping index to character
    blank: index of the <blank> token
    """
    if preds.dim() == 3:
        # Get the indices with maximum probability
        preds = preds.argmax(dim=-1) # (B, T)
        
    decoded_strings = []
    
    for b in range(preds.size(0)):
        pred_indices = preds[b]
        decoded_chars = []
        prev_idx = -1
        
        for idx in pred_indices:
            idx = idx.item()
            # CTC decoding rule: remove repeated characters and remove blanks
            if idx != prev_idx and idx != blank:
                if idx in idx_to_char:
                    decoded_chars.append(idx_to_char[idx])
            prev_idx = idx
            
        decoded_strings.append("".join(decoded_chars))
        
    return decoded_strings

def decode_targets(targets, target_lengths, idx_to_char):
    """
    Decodes the concatenated 1D target tensor back into strings.
    """
    decoded_strings = []
    start = 0
    for length in target_lengths:
        end = start + length
        seq = targets[start:end]
        chars = [idx_to_char[idx.item()] for idx in seq if idx.item() in idx_to_char]
        decoded_strings.append("".join(chars))
        start = end
    return decoded_strings

def edit_distance(s1, s2):
    """
    Computes Levenshtein distance between two strings using DP.
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1],
                                              distances[index1 + 1],
                                              new_distances[-1])))
        distances = new_distances
    return distances[-1]

def calculate_metrics(predictions, targets):
    """
    predictions: list of strings
    targets: list of strings
    Returns CER and Word Accuracy.
    """
    assert len(predictions) == len(targets)
    
    total_distance = 0
    total_chars = 0
    correct_words = 0
    
    for pred, target in zip(predictions, targets):
        # Character Error Rate component
        dist = edit_distance(pred, target)
        total_distance += dist
        total_chars += max(len(target), 1) # Prevent division by zero
        
        # Word Accuracy component
        if pred == target:
            correct_words += 1
            
    cer = (total_distance / total_chars) * 100.0 if total_chars > 0 else 0.0
    word_acc = (correct_words / len(targets)) * 100.0 if len(targets) > 0 else 0.0
    
    return cer, word_acc
