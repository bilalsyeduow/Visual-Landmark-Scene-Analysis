# member3_utils.py
import cv2
import numpy as np
import os

# --- Initialization ---
# These objects are initialized once when the module is imported.
orb = cv2.ORB_create(500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# --- Feature Extraction Functions ---
def extract_orb(img):
    """Extracts ORB descriptors from an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, desc = orb.detectAndCompute(gray, None)
    return desc


def extract_hist(img, bins=(8, 8, 8)):
    """Extracts a color histogram from an image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# --- Similarity Calculation Functions ---
def orb_similarity(d1, d2):
    """Calculates similarity score based on ORB feature matching."""
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return 0
    matches = bf.match(d1, d2)
    if not matches:
        return 0
    # A more robust similarity score than just len(matches)
    similar_points = [m for m in matches if m.distance < 50]  # Adjust threshold as needed
    return len(similar_points)


def hist_similarity(h1, h2):
    """Calculates similarity score based on histogram correlation."""
    return cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_CORREL)


# --- Dataset Loading and Pre-processing ---
def initialize_dataset(folder='dataset'):
    """Loads all images from the dataset folder and pre-computes their features."""
    paths = []
    imgs = []
    if not os.path.exists(folder):
        print(f"Warning: '{folder}' directory not found.")
        return [], [], []

    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            p = os.path.join(folder, fname)
            img = cv2.imread(p)
            if img is not None:
                imgs.append(img)
                paths.append(p)

    orb_descs = [extract_orb(im) for im in imgs]
    hists = [extract_hist(im) for im in imgs]

    return paths, orb_descs, hists


# --- Main Retrieval Function ---
def retrieve_similar_images(query_img, method, db_paths, db_orb_descs, db_hists, topk=3):
    """
    Retrieves the top k similar images from the dataset.
    Args:
        query_img: The user-uploaded image (as a NumPy array).
        method (str): 'orb' or 'hist'.
        db_paths, db_orb_descs, db_hists: Pre-computed dataset features.
        topk (int): Number of similar images to return.
    Returns:
        list: A list of tuples, where each tuple is (path_to_similar_image, score).
    """
    if method == 'orb':
        qf = extract_orb(query_img)
        # Using a lambda to handle None values gracefully during similarity calculation
        sims = [orb_similarity(qf, d) if qf is not None and d is not None else 0 for d in db_orb_descs]
        reverse = True
    else:  # method == 'hist'
        qf = extract_hist(query_img)
        sims = [hist_similarity(qf, h) for h in db_hists]
        reverse = True

    # Get the indices of the top k scores
    idxs = np.argsort(sims)
    if reverse:
        idxs = idxs[::-1]

    top_idxs = idxs[:topk]

    return [(db_paths[i], sims[i]) for i in top_idxs]
