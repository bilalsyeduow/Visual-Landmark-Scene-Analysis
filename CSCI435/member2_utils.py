# member2_utils.py
import cv2
import numpy as np


def assess_image_quality(image_path, blur_thresh=100.0, brightness_thresh=100, day_thresh=100):
    """
    Analyzes an image for blur, brightness, and time of day, returning an annotated image and a report.
    Args:
        image_path (str): Path to the image file.
        blur_thresh (float): Threshold for blur detection.
        brightness_thresh (int): Threshold for brightness.
        day_thresh (int): Threshold for day/night classification.
    Returns:
        tuple: A tuple containing (annotated_image, report_string).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}.")

    # Resize for consistent processing
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Analysis ---
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_status = "Sharp" if lap_var > blur_thresh else "Blurry"

    brightness = np.mean(gray)
    brightness_status = "Good"
    if brightness < brightness_thresh:
        brightness_status = "Too Dark"
    elif brightness > 180:
        brightness_status = "Too Bright"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_mean = np.mean(hsv[:, :, 2])
    time_of_day = "Day" if v_mean > day_thresh else "Night"

    # --- Create Report and Suggestions ---
    messages = [
        f"Time of Day: {time_of_day}",
        f"Blur Status: {blur_status} (Laplacian variance: {lap_var:.2f})",
        f"Brightness: {brightness_status} (Average: {brightness:.2f})"
    ]

    suggestion_messages = []
    annotated_img = img.copy()
    overlay_needed = False

    if blur_status == "Blurry":
        suggestion_messages.append("Suggestion: Image appears blurry. Consider retaking with a stable camera.")

    if brightness_status == "Too Dark":
        suggestion_messages.append("Suggestion: Image is too dark. Try increasing exposure or brightness.")
        # Highlight dark regions in red
        dark_mask = cv2.inRange(gray, 0, 60)
        red_overlay = np.zeros_like(annotated_img)
        red_overlay[:, :] = (0, 0, 255)  # Red in BGR
        annotated_img[dark_mask > 0] = cv2.addWeighted(annotated_img, 0.5, red_overlay, 0.5, 0)[dark_mask > 0]
        overlay_needed = True

    if brightness_status == "Too Bright":
        suggestion_messages.append("Suggestion: Image may be overexposed. Try decreasing exposure.")
        # Highlight bright regions in blue
        bright_mask = cv2.inRange(gray, 220, 255)
        blue_overlay = np.zeros_like(annotated_img)
        blue_overlay[:, :] = (255, 0, 0)  # Blue in BGR
        annotated_img[bright_mask > 0] = cv2.addWeighted(annotated_img, 0.5, blue_overlay, 0.5, 0)[bright_mask > 0]
        overlay_needed = True

    # Combine messages for the final report
    report_list = ["**Image Quality Report:**"] + ["🔹 " + msg for msg in messages]
    if suggestion_messages:
        report_list.append("\n**Suggestions:**")
        report_list.extend(["🔸 " + msg for msg in suggestion_messages])

    report_string = "\n".join(report_list)

    # Convert the final image to RGB for display in Streamlit
    display_image = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    return display_image, report_string
