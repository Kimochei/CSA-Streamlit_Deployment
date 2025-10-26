import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# --- App Configuration ---
st.set_page_config(
    page_title="GI Disease Detection | YOLOv12",
    page_icon="🩺",
    layout="wide"
)

# --- App Title and Description ---
st.title("Gastrointestinal Disease Detection using YOLOv12")
st.markdown("""
    Upload an endoscopic image to detect and segment potential diseases like polyps, esophagitis, or ulcerative colitis.
    The model will draw a colored mask over the detected area and a bounding box with the confidence score.
""")

# --- Model Loading ---
# Use st.cache_resource to load the model only once, making the app faster.
@st.cache_resource
def load_model(model_path):
    """Loads the YOLOv12 model from the specified path."""
    model = YOLO(model_path)
    return model

try:
    model = load_model("best.pt")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Main App Logic ---
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Original Uploaded Image", use_column_width=True)

with col2:
    st.header("Detection Results")
    if uploaded_file:
        # Show a spinner while processing
        with st.spinner("Processing image..."):
            # Convert PIL Image to OpenCV format (NumPy array)
            img_cv = np.array(original_image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            # Run inference on the image
            results = model(img_cv)
            
            # --- Draw Predictions on the Image ---
            overlay = img_cv.copy()
            for r in results:
                # Draw segmentation masks
                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy()
                    for i, mask in enumerate(masks):
                        # Get a random color for each class instance
                        color = np.random.randint(50, 255, (3,), dtype=np.uint8)
                        
                        # Apply the mask
                        overlay[mask > 0] = color

                # Draw bounding boxes and labels
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                class_names = model.names

                for box, score, cls_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{class_names[cls_id]} {score:.2f}"
                    box_color = (0, 255, 0) # Green for the box
                    
                    # Draw the bounding box
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Put the label above the box
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img_cv, (x1, y1 - h - 10), (x1 + w, y1 - 10), box_color, -1)
                    cv2.putText(img_cv, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Blend the overlay with the original image for a transparent mask effect
            processed_image = cv2.addWeighted(overlay, 0.5, img_cv, 0.5, 0)

            # Convert BGR back to RGB for display in Streamlit
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            st.image(processed_image_rgb, caption="Processed Image with Detections", use_column_width=True)
    else:
        st.info("Please upload an image to see the detection results.")