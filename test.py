import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from vocab import build_vocab
from dataset import GujaratiDataset, collate_fn
from model import CRNN
from engine import evaluate

def test_model():
    print(f"Using device: {DEVICE}")
    
    # 1. Build Vocabulary from TRAIN_GT to ensure mapping consistency
    print("Loading vocabulary...")
    char_to_idx, idx_to_char = build_vocab(TRAIN_GT)
    vocab_size = len(char_to_idx)
    
    # 2. Setup Test Dataset & DataLoader
    print("Setting up test dataset...")
    test_dataset = GujaratiDataset(TEST_GT, TEST_DIR, char_to_idx)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0 
    )
    
    # 3. Initialize Model and Load Weights
    print("\nInitializing model and loading 'best_crnn_model.pth'...")
    model = CRNN(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("best_crnn_model.pth", map_location=DEVICE))
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("Error: 'best_crnn_model.pth' not found. Ensure you have trained the model first.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # 4. Evaluate on Test Set
    print("\nStarting evaluation on the Test set...")
    test_loss, test_cer, test_word_acc = evaluate(model, test_loader, criterion, idx_to_char, DEVICE)
    
    print("\n" + "="*40)
    print("🏆 FINAL TEST RESULTS 🏆")
    print("="*40)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Character Error Rate (CER): {test_cer:.2f}%")
    print(f"Test Word Accuracy: {test_word_acc:.2f}%")
    print("="*40 + "\n")
    
    # 5. Visually inspect 10 random test images
    print("Visualizing 10 random test samples...")
    import random
    import matplotlib.pyplot as plt
    from utils import greedy_decode
    
    model.eval()
    
    # Pick 10 random indices from the test dataset
    indices = random.sample(range(len(test_dataset)), min(10, len(test_dataset)))
    
    # Set up a grid of 5 rows x 2 columns
    fig, axes = plt.subplots(5, 2, figsize=(10, 12))
    axes = axes.flatten()
    
    # Matplotlib usually lacks native support for complex Indic scripts (Gujarati) dynamically
    # So we'll print cleanly to the terminal and display the visuals on screen
    print("--- 10 Random Samples ---")
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Dataset directly returns (image_tensor, target_label_indices, unpadded_width_int)
            sample = test_dataset[idx]
            img_tensor = sample[0].unsqueeze(0).to(DEVICE) # (1, 1, H, W)
            target_indices = sample[1]
            
            # Predict
            outputs = model(img_tensor) # (1, W/4, C)
            pred_string = greedy_decode(outputs, idx_to_char, blank=0)[0]
            
            # Decode True Label precisely
            true_string = "".join([idx_to_char[c] for c in target_indices if c in idx_to_char])
            
            # Output in terminal (where Gujarati fonts securely render properly)
            print(f"Sample {i+1}:")
            print(f"  True: {true_string}")
            print(f"  Pred: {pred_string}\n")
            
            # Plot the literal image
            img_np = sample[0].squeeze(0).numpy() # (H, W)
            ax = axes[i]
            ax.imshow(img_np, cmap='gray')
            ax.set_title(f"True vs Pred\n(Check terminal for Gujarati Text)", fontsize=10)
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_model()
