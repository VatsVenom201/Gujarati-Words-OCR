import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from config import *
from vocab import build_vocab
from model import CRNN
from utils import greedy_decode

def load_model():
    char_to_idx, idx_to_char = build_vocab(TRAIN_GT)
    vocab_size = len(char_to_idx)
    
    model = CRNN(vocab_size=vocab_size, hidden_size=HIDDEN_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load("best_crnn_model.pth", map_location=DEVICE, weights_only=True))
        model.eval()
        return model, char_to_idx, idx_to_char
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def segment_lines_and_words(image):
    """
    Given a BGR or Grayscale OpenCV image, 
    returns a list of lines, where each line is a list of word dictionaries:
    [{'box': (x, y, w, h), 'image': cropped_np_array}, ...]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Adaptive thresholding (inverse to get white text on black background)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 9)

    # 1. Line Segmentation
    # Dilate horizontally heavily to connect all words in a line
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
    dilated_lines = cv2.dilate(thresh, kernel_line, iterations=2)
    
    line_contours, _ = cv2.findContours(dilated_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort line bounding boxes top-to-bottom
    line_boxes = [cv2.boundingRect(c) for c in line_contours if cv2.boundingRect(c)[2] > 20 and cv2.boundingRect(c)[3] > 10]
    line_boxes = sorted(line_boxes, key=lambda b: b[1]) # Sort by Y
    
    all_lines = []
    
    for lx, ly, lw, lh in line_boxes:
        # Increase line height slightly to not cut off top/bottom matras
        pad = 8
        ly_p = max(0, ly - pad)
        lh_p = lh + pad*2
        line_thresh = thresh[ly_p:ly_p+lh_p, lx:lx+lw]
        line_gray = gray[ly_p:ly_p+lh_p, lx:lx+lw]
        
        # 2. Word Segmentation within the individual line
        # Dilate slightly to just connect characters of a single word
        kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        dilated_words = cv2.dilate(line_thresh, kernel_word, iterations=1)
        
        word_contours, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter super small contours (noise)
        word_boxes = [cv2.boundingRect(c) for c in word_contours if cv2.boundingRect(c)[2] > 10 and cv2.boundingRect(c)[3] > 10]
        # Sort words left-to-right securely
        word_boxes = sorted(word_boxes, key=lambda b: b[0]) # Sort by X
        
        current_line_words = []
        for wx, wy, ww, wh in word_boxes:
            # Crop exactly from the non-thresholded grayscale original line image
            word_crop = line_gray[wy:wy+wh, wx:wx+ww]
            
            # Map coordinates back to the global original image bounds for plotting UI
            global_x = lx + wx
            global_y = ly_p + wy
            
            current_line_words.append({
                'box': (global_x, global_y, ww, wh),
                'image': word_crop
            })
            
        if current_line_words:
            all_lines.append(current_line_words)
            
    return all_lines

def preprocess_word(word_img):
    """ Matches exact training transformations defined in GujaratiDataset natively """
    pil_img = Image.fromarray(word_img)
    
    w, h = pil_img.size
    aspect_ratio = w / h
    new_w = int(IMG_HEIGHT * aspect_ratio)
    
    if new_w > MAX_WIDTH:
        new_w = MAX_WIDTH
    new_w = max(new_w, 1)
    
    pil_img = pil_img.resize((new_w, IMG_HEIGHT), Image.Resampling.BILINEAR)
    
    transform = transforms.ToTensor()
    tensor = transform(pil_img) # (1, H, W)
    
    # Pad to minimum width to avoid CNN downsampling to 0 length
    c, h, w = tensor.shape
    if w < 4:
        pad_w = 4 - w
        # Pad with 0s (black, common for background representation in CNN inputs)
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, 0), "constant", 0)
        
    tensor = tensor.unsqueeze(0) # Batch representation: (1, 1, H, W)
    return tensor

def predict_word(model, idx_to_char, word_tensor):
    word_tensor = word_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(word_tensor) # (1, T, C)
        pred_string = greedy_decode(outputs, idx_to_char, blank=0)[0]
    return pred_string

def run_pipeline(image, model, idx_to_char):
    """
    Drives entire paragraph processing pipeline sequentially.
    Returns: extracted text blocks parsed by newline, and an annotated OpenCV mapping image.
    """
    lines = segment_lines_and_words(image)
    
    output_image = image.copy()
    if len(output_image.shape) == 2:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
        
    full_text = []
    
    for line in lines:
        line_text = []
        for word_info in line:
            x, y, w, h = word_info['box']
            word_crop = word_info['image']
            
            # Formats cropped image to exact PyTorch Network demands identically
            word_tensor = preprocess_word(word_crop)
            pred = predict_word(model, idx_to_char, word_tensor)
            
            if pred: # Ignore structurally blank predictions if noise tricked bounding box
                line_text.append(pred)
            
            # Draw highly visible bounding box directly mapped on original file for UI
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        full_text.append(" ".join(line_text))
        
    final_string = "\n".join(full_text)
    return final_string, output_image
