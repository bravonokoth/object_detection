import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io

# Page config for better look
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("🕵️ YOLOv8 Object Detection Web App")
st.markdown("Upload an image to detect objects in real-time (80+ classes: people, cars, dogs, etc.)")

# Load model (cached to avoid reloading)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Nano model for speed; swap to "yolov8m.pt" for more accuracy

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Detect button
    if st.button("🔍 Detect Objects", type="primary"):
        with st.spinner("Detecting objects with YOLOv8..."):
            # Convert PIL to numpy for YOLO
            img_array = np.array(image)
            
            # Run inference
            results = model(img_array)
            
            # Get annotated image (with boxes, labels, confidences)
            annotated_img = results[0].plot()
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            result_image = Image.fromarray(annotated_img)

        # Display results side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Before", use_column_width=True)
        with col2:
            st.image(result_image, caption="After Detection", use_column_width=True)

        # Show detection details
        st.subheader("Detected Objects")
        detections = results[0].boxes
        if len(detections) > 0:
            for i, box in enumerate(detections):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                st.write(f"**{label}** (Confidence: {conf:.2f})")
        else:
            st.write("No objects detected!")

        # Download button for annotated image
        buf = io.BytesIO()
        result_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Detected Image",
            data=byte_im,
            file_name="detected_objects.jpg",
            mime="image/jpeg"
        )

# Footer
st.markdown("---")
