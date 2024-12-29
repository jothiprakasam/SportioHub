import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image_path):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    results = pose.process(image_array)

    keypoints = {}
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = (landmark.x, landmark.y, landmark.z)
    return keypoints, image_array

def calculate_similarity(input_keypoints, validation_keypoints):
    distances = []
    for i in input_keypoints:
        if i in validation_keypoints:
            input_point = np.array(input_keypoints[i])
            validation_point = np.array(validation_keypoints[i])
            distance = np.linalg.norm(input_point - validation_point)
            distances.append(distance)
    return np.mean(distances) if distances else float('inf')

def identify_deviations(input_keypoints, validation_keypoints, threshold=0.05):
    deviations = {}
    for i in input_keypoints:
        if i in validation_keypoints:
            distance = np.linalg.norm(np.array(input_keypoints[i]) - np.array(validation_keypoints[i]))
            if distance > threshold:
                deviations[i] = distance
    return deviations

def suggest_corrections(deviations):
    corrections = {
        11: "Adjust your left shoulder position.",
        12: "Adjust your right shoulder position.",
        13: "Lower your left elbow for better alignment.",
        14: "Lower your right elbow for better alignment.",
        23: "Move your left hip slightly forward.",
        24: "Align your right hip with the crease.",
    }
    feedback = []
    for joint, _ in deviations.items():
        if joint in corrections:
            feedback.append(corrections[joint])
    return feedback

def draw_feedback(image, deviations, keypoints, output_path):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for joint, _ in deviations.items():
        if joint in keypoints:
            x, y, _ = keypoints[joint]
            x, y = int(x * image.shape[1]), int(y * image.shape[0])
            ax.scatter(x, y, color='red', s=100)
            ax.text(x, y - 10, "Deviation", color='red', fontsize=12)

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
