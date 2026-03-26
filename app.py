import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import pipeline architecture components directly from backend
from ocr_pipeline import load_model, run_pipeline

st.set_page_config(page_title="Gujarati OCR Architecture System", layout="wide")

st.title("Gujarati Handwritten Region OCR ✍️")
st.markdown("Upload an image containing handwritten Gujarati paragraphs. The backend pipeline maps OpenCV CV2 dynamic contours and sequences cropped components into our fully trained CRNN-CTC model to extract clean text translations natively.")

# Initialize strictly locally cached models avoiding constant memory dumping
@st.cache_resource
def get_model():
    return load_model()

# Bind logic explicitly bridging
model, char_to_idx, idx_to_char = get_model()

if model is None:
    st.error("Failed to load model. Be sure `best_crnn_model.pth` is available in the root directory where the app is being run.")
else:
    st.success("Trained CRNN Model attached successfully into memory boundary!")

uploaded_file = st.file_uploader("Upload Image (JPG / PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read backend image buffers directly mapped into matrix memory dynamically
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1) # BGR
    
    st.subheader("Original Image Input")
    st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), use_container_width=False)
    
    if st.button("Run OCR Extraction Engine", type="primary"):
        with st.spinner("Segmenting document lines dynamically & generating precise GPU word representations natively..."):
            extracted_text, drawn_img = run_pipeline(opencv_image, model, idx_to_char)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Extracted Gujarati Translation")
                if extracted_text.strip():
                    st.text_area("Predicted Text", extracted_text, height=450)
                else:
                    st.warning("No structured contours successfully mapped locally inside network parameters.")
                    
            with col2:
                st.subheader("Segmented Words Boundary Map")
                st.image(cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Green boundaries uniquely trace identified components individually mapped against PyTorch network requirements cleanly.")
