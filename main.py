import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from vocab import build_vocab
from dataset import GujaratiDataset, collate_fn
from model import CRNN
from engine import train_one_epoch, evaluate

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Build Vocabulary
    char_to_idx, idx_to_char = build_vocab(TRAIN_GT)
    vocab_size = len(char_to_idx)
    
    # 2. Datasets & DataLoaders
    print("Setting up datasets...")
    train_dataset = GujaratiDataset(TRAIN_GT, TRAIN_DIR, char_to_idx)
    val_dataset = GujaratiDataset(VAL_GT, VAL_DIR, char_to_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0 # Keep at 0 for safe execution on Windows without freezing dataset threads
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 3. Model, Loss, Optimizer
    print("Initializing model...")
    model = CRNN(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
    
    # zero_infinity=True helps prevent crashes if labels are longer than predictions for an unlikely sequence
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 4. Training Loop
    num_epochs = 20 # Increased max epochs since we now have early stopping
    print("Starting training...")
    
    best_cer = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_cer, val_word_acc = evaluate(model, val_loader, criterion, idx_to_char, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val CER: {val_cer:.2f}% | Val Word Accuracy: {val_word_acc:.2f}%")
        
        if val_cer < best_cer:
            best_cer = val_cer
            patience_counter = 0
            torch.save(model.state_dict(), "best_crnn_model.pth")
            print(f"🌟 Best model saved! (CER improved to {best_cer:.2f}%)")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement in CER. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch} epochs due to no improvement in validation CER.")
            break

if __name__ == "__main__":
    main()
