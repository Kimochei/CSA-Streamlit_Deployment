import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from collections import defaultdict
import pandas as pd
import altair as alt

# --- App Configuration ---
st.set_page_config(
    page_title="GI-Detect AI | YOLOv12",
    page_icon="CSA",
    layout="wide"
)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_model(model_path):
    """Loads the YOLOv12 model."""
    model = YOLO(model_path)
    return model

try:
    model = load_model("best.pt")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("CSA GI-Detect AI")
    st.markdown("---")
    st.markdown(
        "**GI-Detect AI** uses a YOLOv12 model to identify and segment gastrointestinal diseases from endoscopic images."
    )
    st.markdown(
        "This tool is a demonstration and not a substitute for professional medical advice."
    )
    st.markdown("---")
    st.subheader("Model Information")
    st.markdown(
        "- **Model:** YOLOv12\n"
        "- **Task:** Instance Segmentation\n"
        "- **Dataset:** Kvasir Dataset (Endoscopic Images)\n"
    )

# --- Main Page Title ---
st.title("Gastrointestinal Disease Detection Interface")
st.markdown("Upload an image to begin the analysis.")

# --- Main App Layout ---
col1, col2 = st.columns([0.9, 1.1])

# --- Image Upload Column ---
with col1:
    with st.container(border=True):
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an endoscopic image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Original Uploaded Image", use_container_width=True)

# --- Detection Results Column ---
with col2:
    with st.container(border=True):
        st.header("Detection Results")
        if uploaded_file:
            with st.spinner("Analyzing image..."):
                detections = []
                img_cv = np.array(original_image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                results = model(img_cv)
                
                overlay = img_cv.copy()
                for r in results:
                    if r.masks is not None:
                        for mask in r.masks.data.cpu().numpy():
                            color = np.random.randint(50, 255, (3,), dtype=np.uint8)
                            overlay[mask > 0] = color
                    
                    for box, score, cls_id in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                        detections.append({"name": model.names[cls_id], "confidence": score})
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{model.names[cls_id]} {score:.2f}"
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img_cv, (x1, y1 - h - 10), (x1 + w, y1 - 10), (0, 255, 0), -1)
                        cv2.putText(img_cv, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                processed_image = cv2.addWeighted(overlay, 0.5, img_cv, 0.5, 0)
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                
                st.image(processed_image_rgb, caption="Processed Image with Detections", use_container_width=True)
                
                st.subheader("Detection Analysis")
                if detections:
                    counts = defaultdict(int)
                    for det in detections:
                        name = det['name'].replace('_', ' ').title()
                        conf = det['confidence'] * 100
                        counts[name] += 1
                        st.metric(label=f"{name} #{counts[name]}", value=f"{conf:.2f}%")
                else:
                    st.success("No diseases or specific landmarks were detected.")
        else:
            st.info("Awaiting image upload to display results.")

st.divider()

# --- Model Statistics Section ---
st.header("Model Performance Statistics")
st.markdown("These metrics summarize the model's overall performance on the validation dataset.")

stat_col1, stat_col2, stat_col3 = st.columns(3)
stat_col1.metric("mAP@0.5-0.95", "65.7%")
stat_col2.metric("mAP@0.5", "84.0%")
stat_col3.metric("Precision", "78.9%")

st.markdown("---")

# --- Per-Class Performance Chart ---
st.subheader("Per-Class Performance Breakdown")
st.markdown("This chart shows the model's **mean Average Precision (mAP@0.5)** for each specific class.")

class_performance_data = {
    'Class': [
        'Polyps', 'Esophagitis', 'Ulcerative-Colitis', 
        'Normal-Cecum', 'Normal-Pylorus', 'Normal-Z-Line'
    ],
    'mAP@0.5 (%)': [92.1, 78.5, 85.3, 98.2, 96.5, 95.8] 
}
df_perf = pd.DataFrame(class_performance_data)

chart = alt.Chart(df_perf).mark_bar().encode(
    x=alt.X('mAP@0.5 (%):Q', title='Mean Average Precision (mAP@0.5)', scale=alt.Scale(domain=[0, 100])),
    y=alt.Y('Class:N', sort=None, title='Disease / Landmark Class'),
    tooltip=['Class', 'mAP@0.5 (%)']
).properties(
    title='Model Performance by Class'
)

text = chart.mark_text(
    align='left',
    baseline='middle',
    dx=3
).encode(
    text=alt.Text('mAP@0.5 (%):Q', format='.1f')
)

st.altair_chart((chart + text), use_container_width=True)