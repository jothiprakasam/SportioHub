import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(frame):
    """Extract keypoints from a video frame using MediaPipe."""
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
        return np.array(keypoints), results.pose_landmarks
    return None, None


def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def overlay_analysis(frame, pose_landmarks, text=""):
    """Draw pose landmarks and overlay text on the frame."""
    if pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if text:
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def generate_feedback(angle_diff):
    """Generate feedback based on the angle difference."""
    if angle_diff > 10:
        return "Recommendation: Adjust your elbow angle to be closer to the reference."
    elif angle_diff > 5:
        return "Tip: Your elbow is slightly off. Try to match the reference."
    else:
        return "Good form! Your elbow angle is almost perfect."


def analyze_videos(default_video, user_video, output_video):
    """Analyze two videos frame by frame and save combined output."""
    cap_default = cv2.VideoCapture(default_video)
    cap_user = cv2.VideoCapture(user_video)

    # Get video properties
    frame_width = int(cap_default.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_default.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_default.get(cv2.CAP_PROP_FPS))

    # Ensure both videos have the same height for side-by-side comparison
    frame_user_width = int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_user_height = int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_user_height != frame_height:
        frame_user_width = int(frame_width * (frame_user_height / frame_height))

    # Video Writer for output
    output_width = frame_user_width
    output_height = frame_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))

    while cap_default.isOpened() and cap_user.isOpened():
        ret_default, frame_default = cap_default.read()
        ret_user, frame_user = cap_user.read()

        if not ret_default or not ret_user:
            break

        # Resize the user frame to match the height of the default frame
        frame_user = cv2.resize(frame_user, (frame_user_width, frame_height))

        # Extract keypoints and landmarks
        default_keypoints, default_landmarks = extract_keypoints(frame_default)
        user_keypoints, user_landmarks = extract_keypoints(frame_user)

        # Compare right elbow angles (example)
        comparison_text = ""
        feedback_text = ""
        if default_keypoints is not None and user_keypoints is not None:
            default_angle = calculate_angle(default_keypoints[12], default_keypoints[14], default_keypoints[16])
            user_angle = calculate_angle(user_keypoints[12], user_keypoints[14], user_keypoints[16])
            angle_diff = abs(default_angle - user_angle)
            comparison_text = f"Right Elbow Angle Diff: {angle_diff:.2f}°"

            # Generate feedback based on the angle difference
            feedback_text = generate_feedback(angle_diff)

            # Output the feedback in the console (run-time screen)
            print(f"Frame Analysis: {comparison_text}")
            print(f"Feedback: {feedback_text}")
            print("-" * 50)  # Separator for clarity

        # Overlay analysis on user frame
        frame_user = overlay_analysis(frame_user, user_landmarks, "User Shot")

        # Display comparison text and feedback on user side
        cv2.putText(frame_user, comparison_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_user, feedback_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Write the user video frame with analysis to the output video
        out.write(frame_user)

        # Optional: Display the frame during processing
        cv2.imshow('User Shot Analysis', frame_user)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap_default.release()
    cap_user.release()
    out.release()
    cv2.destroyAllWindows()


# Example Usage
default_video_path = r"C:\Users\jothi\projects\sportio2\project\default.mp4"  # Path to the default video
user_video_path = r"C:\Users\jothi\projects\sportio2\project\comparison.mp4"  # Path to the user video
output_video_path = r"C:\Users\Balaji\Videos\cricket videos\cover_drive\comparison_output.mp4"  # Path for the output video

analyze_videos(default_video_path, user_video_path, output_video_path)
