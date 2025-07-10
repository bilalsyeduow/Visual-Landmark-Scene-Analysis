# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# --- Custom Module Imports ---
# These files must be in the same directory as app.py
from member1_utils import detect_landmarks_and_faces
from member2_utils import assess_image_quality
from member3_utils import retrieve_similar_images, initialize_dataset

# --- Page Configuration ---
st.set_page_config(
    page_title="Visual Landmark & Scene Analysis",
    page_icon="🖼️",
    layout="wide"
)

# --- App Title ---
st.title("CSCI435: Visual Landmark & Scene Analysis")
st.write("Integrated prototype with all functional modules.")

# --- File Paths and Session State ---
UPLOADS_DIR = "uploads"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

if 'uploaded_image_path' not in st.session_state:
    st.session_state.uploaded_image_path = None


# --- Helper Function ---
def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to the 'uploads' directory."""
    if uploaded_file is not None:
        # Use PIL to open the image and save it in a consistent format (e.g., PNG)
        # This avoids potential issues with different image formats.
        image = Image.open(uploaded_file)
        file_path = os.path.join(UPLOADS_DIR, "uploaded_image.png")
        image.save(file_path)
        return file_path
    return None


# --- Sidebar for Navigation and Controls ---
st.sidebar.header("Project Modules")
app_mode = st.sidebar.selectbox(
    "Choose a module to test:",
    [
        "Home",
        "1. Landmark and Face Recognition",
        "2. Time-of-Day & Quality",
        "3. Similarity Retrieval",
        "4. Image Annotation Tool"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Image Upload")
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.session_state.uploaded_image_path = save_uploaded_file(uploaded_file)
    st.sidebar.image(st.session_state.uploaded_image_path, caption="Uploaded Image", use_container_width=True)


# --- Initialize Datasets (Caching for efficiency) ---
@st.cache_resource
def load_similarity_dataset():
    """Loads and caches the dataset for Member 3's module."""
    try:
        paths, orb_descs, hists = initialize_dataset()
        return paths, orb_descs, hists
    except Exception as e:
        st.error(f"Error initializing similarity dataset: {e}")
        st.info("Please make sure you have a 'dataset' folder with images in it.")
        return [], [], []


db_paths, db_orb_descs, db_hists = load_similarity_dataset()

# --- Main Page Content ---
if app_mode == "Home":
    st.header("Welcome to the Integrated Project!")
    st.markdown("""
    This application combines all the modules developed by the team.
    1. **Upload an image** using the sidebar to activate the modules.
    2. **Select a module** from the dropdown to see the result.
    """)
    if st.session_state.uploaded_image_path:
        st.subheader("Current Image:")
        image = Image.open(st.session_state.uploaded_image_path)
        st.image(image, caption="The uploaded image will be used in all modules.", use_container_width=True)
    else:
        st.info("Please upload an image to begin.")


# --- Module 1: Landmark and Face Recognition ---
elif app_mode == "1. Landmark and Face Recognition":
    st.header("Landmark and Face Recognition (Member 1)")
    if st.session_state.uploaded_image_path:
        if st.button("Detect Landmarks and Faces"):
            with st.spinner("Processing for landmarks and faces..."):
                try:
                    processed_image = detect_landmarks_and_faces(st.session_state.uploaded_image_path)
                    st.image(processed_image, caption="Detection Results", use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.info("Please ensure you have a 'landmarks' folder with landmark images.")
    else:
        st.warning("Please upload an image first.")


# --- Module 2: Time-of-Day & Quality ---
elif app_mode == "2. Time-of-Day & Quality":
    st.header("Time-of-Day Classification and Image Quality (Member 2)")
    if st.session_state.uploaded_image_path:
        if st.button("Analyze Image"):
            with st.spinner("Assessing image quality..."):
                try:
                    annotated_image, report = assess_image_quality(st.session_state.uploaded_image_path)
                    st.subheader("Analysis Report")
                    st.markdown(report.replace("🔹", "-"))
                    st.subheader("Visual Suggestions")
                    st.image(annotated_image, caption="Highlighted Areas for Improvement", use_container_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image first.")


# --- Module 3: Similarity Retrieval ---
elif app_mode == "3. Similarity Retrieval":
    st.header("Similarity Retrieval (Member 3)")
    if st.session_state.uploaded_image_path:
        similarity_method = st.selectbox("Choose similarity criteria:", ('orb', 'hist'))
        if st.button("Find Similar Images"):
            if not db_paths:
                st.error("The similarity dataset is empty. Please add images to the 'dataset' folder and refresh.")
            else:
                with st.spinner(f"Retrieving images based on '{similarity_method}' similarity..."):
                    try:
                        query_img = cv2.imread(st.session_state.uploaded_image_path)
                        results = retrieve_similar_images(query_img, similarity_method, db_paths, db_orb_descs,
                                                          db_hists, topk=3)

                        st.subheader("Top 3 Similar Images")
                        cols = st.columns(3)
                        for i, (path, score) in enumerate(results):
                            with cols[i]:
                                image = Image.open(path)
                                st.image(image, caption=f"{os.path.basename(path)}\nScore: {score:.2f}")
                    except Exception as e:
                        st.error(f"An error occurred during retrieval: {e}")
    else:
        st.warning("Please upload an image first.")


# --- Module 4: Image Annotation Tool ---
elif app_mode == "4. Image Annotation Tool":
    st.header("Image Annotation Tool (Member 4)")
    st.write("Draw directly on the image. Annotation data is shown below.")
    if st.session_state.uploaded_image_path:
        bg_image = Image.open(st.session_state.uploaded_image_path)

        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color: ", "#ff0000")
        drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))

        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_image=bg_image,
            update_streamlit=True,
            height=bg_image.height,
            width=bg_image.width,
            drawing_mode=drawing_mode,
            key="canvas",
        )

        if canvas_result.json_data is not None:
            st.subheader("Annotation Data")
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            if not objects.empty:
                st.dataframe(objects)
    else:
        st.warning("Please upload an image first to use the annotation tool.")

