import sys
import tkinter as tk
from tkinter import filedialog
import cv2

# Import the necessary functions from your existing OCR pipeline
from ocr_pipeline import load_model, preprocess_word, predict_word

def crop_to_text(gray_img):
    """Automatically crops away all blank white space around the word."""
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 9)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return gray_img
        
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    valid_contours_found = False
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 2 and h > 2: # filter tiny noise
            valid_contours_found = True
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)
            
    if not valid_contours_found: return gray_img
        
    # Add 4px padding so it isn't literally touching the border
    pad = 4
    y_min = max(0, y_min - pad)
    y_max = min(gray_img.shape[0], y_max + pad)
    x_min = max(0, x_min - pad)
    x_max = min(gray_img.shape[1], x_max + pad)
    
    return gray_img[y_min:y_max, x_min:x_max]

def normalize_image(img):
    """Advanced normalization: Denoise, CLAHE contrast enhancement, and scale."""
    # Step 1: Denoise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 2: CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Note: We keep it as uint8 here so it works with the existing 
    # Image.fromarray(word_img) inside preprocess_word()
    return img

def main():
    # Hide the main root Tkinter window so only the dialog shows
    root = tk.Tk()
    root.withdraw()
    # Ensure the window appears in front of others
    root.attributes('-topmost', True)

    print("Opening file dialog to select an image...")
    file_path = filedialog.askopenfilename(
        title="Select a Single Word Gujarati Image",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        print("No file selected. Exiting...")
        sys.exit(0)

    print(f"\nSelected Image: {file_path}")

    # Load the image using OpenCV in Grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read the image. It might be corrupted or an unsupported format.")
        sys.exit(1)

    # Advanced Image Normalization
    print("Applying advanced image normalization (Denoise + CLAHE)...")
    image = normalize_image(image)

    print("Loading CRNN model & vocab...")
    model, _, idx_to_char = load_model()
    
    if model is None:
        print("Failed to load 'best_crnn_model.pth'. Check if the file exists.")
        sys.exit(1)

    print("Auto-cropping whitespace from the picture...")
    cropped_image = crop_to_text(image)

    print("Preprocessing image to tensor format...")
    # Preprocess the grayscale word image to (1, 1, 32, W)
    word_tensor = preprocess_word(cropped_image)
    
    # Save the tensor as an image so the user can manually inspect what the model saw
    from torchvision.utils import save_image
    save_image(word_tensor, "debug_word_input.png")
    print("\n[!] Saved 'debug_word_input.png' to your folder.")
    print("    -> Check this image! If it's pure black, blurry, or too squashed, that's why prediction is empty.")

    print("\nRunning greedy inference...")
    # Retrieve final text prediction string
    prediction = predict_word(model, idx_to_char, word_tensor)
    
    # Print out nicely
    print("=" * 50)
    print(f"Predicted Text:  {prediction}")
    print("=" * 50)

if __name__ == "__main__":
    main()
