import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image_path):
    """
    Extract landmarks from an image path.
    Returns:
        keypoints: dict mapping landmark_id -> (x, y, z, visibility)
        image_bgr: original OpenCV BGR image array
    """
    if not os.path.exists(image_path):
        return None, None

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None, None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None, image_bgr

    keypoints = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        keypoints[idx] = (lm.x, lm.y, lm.z, lm.visibility)

    return keypoints, image_bgr

def calculate_similarity(input_keypoints, validation_keypoints):
    """
    Calculate pose similarity percentage between two sets of keypoints (0% to 100%).
    """
    if not input_keypoints or not validation_keypoints:
        return 0.0

    # Common joints to compare
    joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    distances = []

    for j in joints:
        if j in input_keypoints and j in validation_keypoints:
            p1 = np.array(input_keypoints[j][:2])
            p2 = np.array(validation_keypoints[j][:2])
            dist = np.linalg.norm(p1 - p2)
            distances.append(dist)

    if not distances:
        return 0.0

    mean_dist = np.mean(distances)
    # Convert normalized distance to score out of 100 (smaller dist = higher score)
    score = max(20.0, min(100.0, 100.0 - (mean_dist * 180.0)))
    return round(float(score), 1)

def identify_deviations(input_keypoints, validation_keypoints, threshold=0.08):
    """
    Identify joints that deviate beyond the given threshold from the reference pose.
    """
    deviations = {}
    joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    for j in joints:
        if j in input_keypoints and j in validation_keypoints:
            p1 = np.array(input_keypoints[j][:2])
            p2 = np.array(validation_keypoints[j][:2])
            dist = float(np.linalg.norm(p1 - p2))
            if dist > threshold:
                deviations[j] = round(dist, 3)

    return deviations

def suggest_corrections(deviations):
    """
    Generate sports coaching feedback based on deviating joints.
    """
    corrections = {
        11: "Elevate your left lead shoulder to align with the trajectory.",
        12: "Keep your right shoulder relaxed and non-restricted during the stroke.",
        13: "Raise your lead elbow higher towards mid-off to guide the drive along the turf.",
        14: "Tuck your rear elbow smoothly closer to the body during backlift.",
        15: "Keep top-hand wrist firm and positioned ahead of the bat handle.",
        16: "Loosen bottom-hand grip slightly to maintain delicate control.",
        23: "Position your left hip forward into the line of the delivery.",
        24: "Keep back hip square to stabilize your center of gravity.",
        25: "Flex your lead front knee deeper to lean your weight over the ball.",
        26: "Extend the rear leg securely for a stable hitting base.",
        27: "Align front foot pointing towards cover/extra-cover.",
        28: "Stay on the ball of your back foot to allow smooth weight transfer."
    }

    feedback = []
    for joint in deviations:
        if joint in corrections:
            feedback.append(corrections[joint])

    if not feedback:
        feedback.append("Excellent form! Pose closely aligns with the reference standard.")

    return feedback

def draw_feedback(image_bgr, deviations, keypoints, output_path):
    """
    Draw deviation visual highlights directly on the image and save to output_path.
    """
    annotated = image_bgr.copy()
    h, w, _ = annotated.shape

    # Draw deviations
    for joint, dist in deviations.items():
        if joint in keypoints:
            x, y = int(keypoints[joint][0] * w), int(keypoints[joint][1] * h)
            # Red deviation marker
            cv2.circle(annotated, (x, y), 12, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, (x, y), 16, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Dev: {dist:.2f}", (x + 18, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # Add header banner
    cv2.rectangle(annotated, (15, 15), (320, 60), (20, 20, 20), -1)
    cv2.rectangle(annotated, (15, 15), (320, 60), (145, 71, 255), 2)
    dev_count = len(deviations)
    banner_text = f"Deviations Detected: {dev_count}"
    cv2.putText(annotated, banner_text, (28, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return output_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SportioHub Pose Analysis Tool (Image)")
    parser.add_argument("--input", default="upload.jpeg", help="Input image to analyze")
    parser.add_argument("--ref", default="test.jpeg", help="Reference standard image")
    parser.add_argument("--output", default="static/uploads/output_cli.jpg", help="Output path")
    args = parser.parse_args()

    k_in, img_in = extract_keypoints(args.input)
    k_ref, img_ref = extract_keypoints(args.ref)

    if not k_in or not k_ref:
        print("Pose detection failed on one or both images.")
    else:
        score = calculate_similarity(k_in, k_ref)
        devs = identify_deviations(k_in, k_ref)
        fb = suggest_corrections(devs)
        out = draw_feedback(img_in, devs, k_in, args.output)
        print(f"Similarity Score: {score}%")
        print("Corrections:", fb)
        print(f"Output saved to: {out}")
