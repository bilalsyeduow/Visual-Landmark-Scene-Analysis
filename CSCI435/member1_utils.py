# member1_utils.py
import cv2
import numpy as np
import os

# --- Initialization ---
# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Initialize the ORB detector
orb = cv2.ORB_create()


# --- Load Landmark Data ---
# This function loads landmark images and computes their descriptors.
# It should be called once when the application starts.
def load_landmark_data(folder='landmarks'):
    """Loads images from the landmark folder and computes their ORB features."""
    landmark_images = []
    landmark_descriptors = []
    landmark_keypoints = []
    landmark_names = []

    # Simple mapping from filename to a more readable name.
    # This can be expanded or replaced with a more robust system.
    titles = {
        'burj_khalifa_1.jpg': 'Burj Khalifa', 'burj_khalifa_2.jpg': 'Burj Khalifa',
        'burj_khalifa_3.jpg': 'Burj Khalifa', 'burj_khalifa_4.jpg': 'Burj Khalifa',
        'atlantis.jpg': 'Atlantis', 'atlantis_night.jpg': 'Atlantis The Royal',
        'atlantis_royal.jpg': 'Atlantis The Royal', 'burj_al_arab.jpeg': 'Burj Al Arab',
        'peace_statue.jpeg': 'Peace Statue', 'boat.jpeg': 'Boat',
        'palm.jpg': 'Palm Jumeirah', 'car.jpg': 'Car', 'plane.jpg': 'Airplane',
        'opera.jpg': 'Dubai Opera', 'ain.jpg': 'Ain Dubai'
    }

    if not os.path.exists(folder):
        print(f"Warning: '{folder}' directory not found.")
        return [], [], [], []

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, 0)  # Load in grayscale
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                if des is not None:
                    landmark_images.append(img)
                    landmark_keypoints.append(kp)
                    landmark_descriptors.append(des)
                    landmark_names.append(titles.get(filename, os.path.splitext(filename)[0]))
    return landmark_images, landmark_keypoints, landmark_descriptors, landmark_names


# --- Load the data globally when the module is imported ---
landmark_images, landmark_keypoints, landmark_descriptors, landmark_names = load_landmark_data()


# --- Main Detection Function ---
def detect_landmarks_and_faces(img_path):
    """
    Detects faces and landmarks in the given image.
    Args:
        img_path (str): The path to the user-uploaded image.
    Returns:
        np.array: The image with detections drawn on it.
    """
    if not landmark_images:
        raise FileNotFoundError("Landmark data not loaded. Ensure the 'landmarks' folder is populated.")

    img_color = cv2.imread(img_path)
    if img_color is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # --- Face Detection ---
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_color, 'Face', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # --- Landmark Detection ---
    kp2, des2 = orb.detectAndCompute(img_gray, None)
    if des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # Iterate through each known landmark
        for i, des1 in enumerate(landmark_descriptors):
            # Use knnMatch for better filtering of matches
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply Lowe's ratio test
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            # If enough good matches are found, consider it a detection
            if len(good) > 20:  # This threshold can be adjusted
                src_pts = np.float32([landmark_keypoints[i][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = landmark_images[i].shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    # Draw a bounding box and label
                    img_color = cv2.polylines(img_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(img_color, landmark_names[i], (int(dst[0][0][0]), int(dst[0][0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    break  # Stop after finding the first good match

    # Convert BGR (OpenCV format) to RGB (Streamlit/PIL format)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    return img_rgb
