import torch
import torch.nn as nn
from tqdm import tqdm
from utils import greedy_decode, decode_targets, calculate_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (images, targets, input_lengths, target_lengths) in enumerate(loop):
        images = images.to(device)
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        
        # outputs shape: (B, T, C)
        outputs = model(images)
        
        # Dynamically compute input_lengths based on actual CNN output sequence length
        b, t, c = outputs.size()
        input_lengths = torch.full(size=(b,), fill_value=t, dtype=torch.long).to(device)
        
        # CTC loss expects (T, B, C)
        outputs = outputs.permute(1, 0, 2)
        
        # Apply log_softmax to outputs (CTCLoss requires log probabilities)
        log_probs = nn.functional.log_softmax(outputs, dim=2)
        
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, idx_to_char, device):
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    loop = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, targets, input_lengths, target_lengths in loop:
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            outputs = model(images) # (B, T, C)
            
            # Dynamically compute input_lengths based on actual CNN output sequence length
            b, t, c = outputs.size()
            input_lengths = torch.full(size=(b,), fill_value=t, dtype=torch.long).to(device)
            
            outputs_permuted = outputs.permute(1, 0, 2) # (T, B, C)
            
            log_probs = nn.functional.log_softmax(outputs_permuted, dim=2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # Decode predictions
            pred_strings = greedy_decode(outputs, idx_to_char, blank=0)
            target_strings = decode_targets(targets, target_lengths, idx_to_char)
            
            all_preds.extend(pred_strings)
            all_targets.extend(target_strings)
            
    # Calculate Metrics
    cer, word_acc = calculate_metrics(all_preds, all_targets)
    
    # Print a few examples
    print("\n--- Validation Examples ---")
    for i in range(min(5, len(all_preds))):
        print(f"Target: [{all_targets[i]}] | Pred: [{all_preds[i]}]")
    print("---------------------------\n")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, cer, word_acc
