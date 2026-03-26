import torch
import random
from config import *
from vocab import build_vocab
from dataset import GujaratiDataset
from model import CRNN
from utils import greedy_decode

def random_visual_test(num_samples=10):
    print(f"Using device: {DEVICE}")
    
    # 1. Build Vocabulary
    print("Loading vocabulary...")
    char_to_idx, idx_to_char = build_vocab(TRAIN_GT)
    vocab_size = len(char_to_idx)
    
    # 2. Setup Test Dataset
    print("Loading test dataset...")
    test_dataset = GujaratiDataset(TEST_GT, TEST_DIR, char_to_idx)
    
    if len(test_dataset) == 0:
        print("Test dataset is empty or could not be loaded.")
        return
    
    # 3. Initialize Model and Load Weights
    print("Initializing model...")
    model = CRNN(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("best_crnn_model.pth", map_location=DEVICE))
        print("Weights loaded successfully!\n")
    except FileNotFoundError:
        print("Error: 'best_crnn_model.pth' not found. Make sure you've run training first.")
        return
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    model.eval()
    
    # Pick random indices
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    print("="*40)
    print(f"🎲 Random {num_samples} Predictions 🎲")
    print("="*40)
    
    correct = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # The dataset returns (image_tensor, target_label_indices, unpadded_w)
            sample = test_dataset[idx]
            img_tensor = sample[0].unsqueeze(0).to(DEVICE) # (1, 1, H, W)
            target_indices = sample[1]
            
            # Predict
            outputs = model(img_tensor) # (1, T, C)
            pred_string = greedy_decode(outputs, idx_to_char, blank=0)[0]
            
            # Decode True Label
            true_string = "".join([idx_to_char[c] for c in target_indices if c in idx_to_char])
            
            # Check match visually
            match = "✅ Match" if pred_string == true_string else "❌ Mismatch"
            if pred_string == true_string:
                correct += 1
                
            print(f"Sample {i+1} [Dataset Index: {idx}]")
            print(f" True: {true_string}")
            print(f" Pred: {pred_string}")
            print(f" Result: {match}\n")
            
    print("="*40)
    print(f"Random Sample Accuracy: {correct}/{num_samples} ({(correct/min(num_samples, len(test_dataset)))*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    random_visual_test(10)
