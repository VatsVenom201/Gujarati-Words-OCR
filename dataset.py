import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from config import IMG_HEIGHT, MAX_WIDTH

class GujaratiDataset(Dataset):
    def __init__(self, gt_path, img_dir, char_to_idx):
        self.img_dir = img_dir
        self.char_to_idx = char_to_idx
        self.samples = []
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    rel_img_path = parts[0] # e.g., images/1.jpg
                    label = parts[1]
                    
                    img_name = os.path.basename(rel_img_path)
                    full_img_path = os.path.join(img_dir, "images", img_name)
                    
                    if os.path.exists(full_img_path):
                        self.samples.append((full_img_path, label))
                    else:
                        print(f"Warning: Image not found {full_img_path}, skipping.")
                        
        print(f"Loaded {len(self.samples)} valid samples out of {gt_path}")
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('L', (32, 32), color=0)
            label = ""
            
        w, h = image.size
        
        aspect_ratio = w / h
        new_w = int(IMG_HEIGHT * aspect_ratio)
        
        if new_w > MAX_WIDTH:
            new_w = MAX_WIDTH
            
        new_w = max(new_w, 1)

        image = image.resize((new_w, IMG_HEIGHT), Image.Resampling.BILINEAR)
        image_tensor = self.transform(image) # (1, H, W)

        # map label to sequence of integer indices
        encoded_label = [self.char_to_idx[c] for c in list(label) if c in self.char_to_idx]

        return image_tensor, encoded_label, new_w

def collate_fn(batch):
    images, labels, widths = zip(*batch)
    
    max_w = max(max(widths), 4) # Ensure minimum width of 4 for CNN poolings
    
    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_w = max_w - w
        padding = (0, pad_w, 0, 0) # Pad right
        padded_img = torch.nn.functional.pad(img, padding, "constant", 0)
        padded_images.append(padded_img)
        
    images_tensor = torch.stack(padded_images)
    
    concatenated_labels = []
    target_lengths = []
    
    for label in labels:
        concatenated_labels.extend(label)
        target_lengths.append(len(label))
        
    targets_tensor = torch.tensor(concatenated_labels, dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    
    # CNN downsample factor (width reduction). Assuming CNN divides width by 4.
    cnn_w_reduction = 4
    input_lengths = [max(w // cnn_w_reduction, 1) for w in widths]
    input_lengths_tensor = torch.tensor(input_lengths, dtype=torch.long)
    
    return images_tensor, targets_tensor, input_lengths_tensor, target_lengths_tensor
