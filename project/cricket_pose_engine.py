import os
import cv2
import numpy as np
import mediapipe as mp
import subprocess
import imageio_ffmpeg

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(p1, p2, p3):
    """
    Calculate angle in degrees at vertex p2 between (p1 - p2) and (p3 - p2).
    p1, p2, p3 are (x, y) or (x, y, z) coordinate tuples or arrays.
    """
    a = np.array(p1[:2], dtype=np.float32)
    b = np.array(p2[:2], dtype=np.float32)
    c = np.array(p3[:2], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)

def extract_landmarks(frame, pose_detector):
    """
    Process an RGB frame and extract keypoints dict and landmarks object.
    Returns:
        keypoints: dict mapping landmark index to (x, y, z, visibility)
        landmarks: MediaPipe LandmarkList object
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)

    if not results.pose_landmarks:
        return None, None

    keypoints = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        keypoints[idx] = (lm.x, lm.y, lm.z, lm.visibility)

    return keypoints, results.pose_landmarks

def compute_cricket_angles(keypoints):
    """
    Compute key cricket batting biomechanical angles.
    For right-handed batsman (facing bowler with left side forward):
    - Lead Elbow: Left shoulder (11), Left elbow (13), Left wrist (15)
    - Rear Elbow: Right shoulder (12), Right elbow (14), Right wrist (16)
    - Front Knee: Left hip (23), Left knee (25), Left ankle (27)
    - Rear Knee: Right hip (24), Right knee (26), Right ankle (28)
    - Spine Angle: Trunk lean relative to vertical
    """
    if not keypoints:
        return None

    # Check key joints availability
    required = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    if any(k not in keypoints for k in required):
        return None

    lead_elbow = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
    rear_elbow = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
    front_knee = calculate_angle(keypoints[23], keypoints[25], keypoints[27])
    rear_knee = calculate_angle(keypoints[24], keypoints[26], keypoints[28])

    # Shoulder tilt angle
    l_sh = keypoints[11]
    r_sh = keypoints[12]
    sh_diff_y = l_sh[1] - r_sh[1]
    sh_diff_x = l_sh[0] - r_sh[0]
    shoulder_tilt = float(np.degrees(np.arctan2(sh_diff_y, sh_diff_x)))

    # Head over front knee alignment (horizontal offset normalized)
    nose = keypoints[0]
    front_knee_x = keypoints[25][0]
    head_knee_offset = float(abs(nose[0] - front_knee_x))

    return {
        "lead_elbow": round(lead_elbow, 1),
        "rear_elbow": round(rear_elbow, 1),
        "front_knee": round(front_knee, 1),
        "rear_knee": round(rear_knee, 1),
        "shoulder_tilt": round(shoulder_tilt, 1),
        "head_knee_offset": round(head_knee_offset, 3)
    }

def get_shot_phase(progress):
    """
    Determine cricket batting phase by normalized video progress (0.0 to 1.0).
    """
    if progress < 0.25:
        return "Stance & Setup"
    elif progress < 0.50:
        return "Backlift & Stride"
    elif progress < 0.75:
        return "Downswing & Impact"
    else:
        return "Follow-Through & Balance"

def evaluate_frame_form(user_angles, ref_angles, phase):
    """
    Compare user angles against reference angles and generate
    score and real-time coaching feedback.
    """
    if not user_angles or not ref_angles:
        return 70.0, "Maintain balance and track the ball.", "neutral"

    elbow_diff = abs(user_angles["lead_elbow"] - ref_angles["lead_elbow"])
    knee_diff = abs(user_angles["front_knee"] - ref_angles["front_knee"])
    rear_elbow_diff = abs(user_angles["rear_elbow"] - ref_angles["rear_elbow"])

    # Score calculation (100 minus deviation penalties)
    penalty = (elbow_diff * 0.45) + (knee_diff * 0.35) + (rear_elbow_diff * 0.20)
    score = max(35.0, min(100.0, 100.0 - penalty))

    feedback = ""
    status = "good"

    if elbow_diff > 25:
        status = "warning"
        if user_angles["lead_elbow"] > ref_angles["lead_elbow"]:
            feedback = "Elevate your lead elbow higher towards mid-off!"
        else:
            feedback = "Extend your lead elbow; don't cramp the bat swing."
    elif knee_diff > 25:
        status = "warning"
        if user_angles["front_knee"] > ref_angles["front_knee"]:
            feedback = "Bend your front knee deeper into the drive for balance."
        else:
            feedback = "Avoid collapsing front knee; keep a stable base."
    elif elbow_diff > 12:
        status = "moderate"
        feedback = "Good motion; refine top-hand elbow alignment."
    else:
        status = "good"
        if phase == "Downswing & Impact":
            feedback = "Superb drive! High elbow & head over the ball."
        elif phase == "Follow-Through & Balance":
            feedback = "Clean follow-through with solid balance held."
        else:
            feedback = "Excellent stance and stride mechanics."

    return round(score, 1), feedback, status

def draw_skeleton_with_status(frame, landmarks, status="good", label=""):
    """
    Draw pose skeleton with color based on form status.
    """
    if not landmarks:
        return frame

    h, w, _ = frame.shape

    # Choose color palette based on status
    if status == "good":
        conn_color = (0, 220, 100)   # Vibrant Green
        joint_color = (50, 255, 150)
    elif status == "moderate":
        conn_color = (0, 190, 255)   # Amber / Orange
        joint_color = (50, 220, 255)
    elif status == "warning":
        conn_color = (60, 60, 255)   # Crimson Red
        joint_color = (100, 100, 255)
    else:
        conn_color = (255, 200, 0)   # Cyan / Blue for reference
        joint_color = (255, 240, 100)

    # Connections to draw for cricket batting
    cricket_connections = [
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm (Lead arm)
        (12, 14), (14, 16),  # Right arm (Rear arm)
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25), (25, 27),  # Left leg (Front leg)
        (24, 26), (26, 28),  # Right leg (Rear leg)
    ]

    points = {}
    for idx, lm in enumerate(landmarks.landmark):
        if lm.visibility > 0.35:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[idx] = (cx, cy)

    # Draw lines
    for start_idx, end_idx in cricket_connections:
        if start_idx in points and end_idx in points:
            cv2.line(frame, points[start_idx], points[end_idx], conn_color, 3, cv2.LINE_AA)

    # Draw joint nodes
    for idx, pt in points.items():
        if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            cv2.circle(frame, pt, 6, joint_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

    return frame

def create_side_by_side_frame(ref_frame, user_frame, ref_angles, user_angles, 
                              score, feedback, status, phase, target_h=720):
    """
    Render reference and user frames side-by-side with professional HUD.
    """
    # Resize both frames to have uniform target_h
    ref_h, ref_w = ref_frame.shape[:2]
    user_h, user_w = user_frame.shape[:2]

    new_ref_w = int(ref_w * (target_h / ref_h))
    new_user_w = int(user_w * (target_h / user_h))

    resized_ref = cv2.resize(ref_frame, (new_ref_w, target_h))
    resized_user = cv2.resize(user_frame, (new_user_w, target_h))

    combined_w = new_ref_w + new_user_w
    combined = np.zeros((target_h + 110, combined_w, 3), dtype=np.uint8)

    # Paste frames
    combined[:target_h, :new_ref_w] = resized_ref
    combined[:target_h, new_ref_w:] = resized_user

    # Divider line
    cv2.line(combined, (new_ref_w, 0), (new_ref_w, target_h), (80, 80, 80), 3)

    # Reference Header Badge
    cv2.rectangle(combined, (20, 20), (280, 70), (20, 20, 20), -1)
    cv2.rectangle(combined, (20, 20), (280, 70), (0, 200, 255), 2)
    cv2.putText(combined, "REFERENCE SHOT (PRO)", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)

    # User Header Badge
    user_badge_x = new_ref_w + 20
    cv2.rectangle(combined, (user_badge_x, 20), (user_badge_x + 280, 70), (20, 20, 20), -1)
    badge_col = (0, 220, 100) if status == "good" else ((0, 180, 255) if status == "moderate" else (60, 60, 255))
    cv2.rectangle(combined, (user_badge_x, 20), (user_badge_x + 280, 70), badge_col, 2)
    cv2.putText(combined, "YOUR PERFORMANCE", (user_badge_x + 25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Phase Badge (Top Center)
    phase_text = f"Phase: {phase}"
    text_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    center_x = combined_w // 2
    cv2.rectangle(combined, (center_x - (text_size[0] // 2) - 15, 18), 
                  (center_x + (text_size[0] // 2) + 15, 65), (35, 25, 60), -1)
    cv2.rectangle(combined, (center_x - (text_size[0] // 2) - 15, 18), 
                  (center_x + (text_size[0] // 2) + 15, 65), (145, 71, 255), 2)
    cv2.putText(combined, phase_text, (center_x - (text_size[0] // 2), 48), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 210, 255), 2, cv2.LINE_AA)

    # On-screen metrics on Reference side
    if ref_angles:
        ref_txt = f"Elbow: {ref_angles['lead_elbow']:.0f} | Knee: {ref_angles['front_knee']:.0f}"
        cv2.putText(combined, ref_txt, (35, target_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # On-screen metrics on User side
    if user_angles and ref_angles:
        elbow_diff = user_angles['lead_elbow'] - ref_angles['lead_elbow']
        diff_sign = "+" if elbow_diff >= 0 else ""
        user_txt = f"Elbow: {user_angles['lead_elbow']:.0f} ({diff_sign}{elbow_diff:.0f}) | Knee: {user_angles['front_knee']:.0f}"
        cv2.putText(combined, user_txt, (user_badge_x + 15, target_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # BOTTOM HUD PANEL
    hud_y = target_h
    cv2.rectangle(combined, (0, hud_y), (combined_w, hud_y + 110), (18, 18, 22), -1)
    cv2.line(combined, (0, hud_y), (combined_w, hud_y), (145, 71, 255), 2)

    # Score Meter
    score_txt = f"Form Score: {score:.1f}%"
    cv2.putText(combined, score_txt, (30, hud_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # Score progress bar
    bar_x = 30
    bar_y = hud_y + 58
    bar_w = 260
    bar_h = 16
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 55), -1)
    fill_w = int(bar_w * (score / 100.0))
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), badge_col, -1)
    cv2.rectangle(combined, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 140), 1)

    # Live Coaching Feedback
    cv2.putText(combined, "COACH'S TIP:", (340, hud_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (145, 71, 255), 2, cv2.LINE_AA)
    cv2.putText(combined, feedback, (340, hud_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    return combined

def transcode_to_h264(input_video_path, output_video_path):
    """
    Use imageio-ffmpeg binary to transcode video to standard H.264 (avc1/yuv420p).
    Guarantees full compatibility across modern browsers.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_video_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        output_video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg transcode error: {result.stderr.decode('utf-8', errors='ignore')}")
    return output_video_path

def analyze_cricket_videos(default_video_path, user_video_path, output_video_path, 
                           snapshots_dir=None, max_frames=None, progress_callback=None):
    """
    Analyze reference vs user cricket shot videos frame-by-frame.
    Extracts MediaPipe landmarks, computes angles, provides real-time coaching tips,
    creates side-by-side video, and returns a comprehensive analysis report.
    """
    if not os.path.exists(default_video_path):
        raise FileNotFoundError(f"Reference video not found at: {default_video_path}")
    if not os.path.exists(user_video_path):
        raise FileNotFoundError(f"User video not found at: {user_video_path}")

    cap_ref = cv2.VideoCapture(default_video_path)
    cap_user = cv2.VideoCapture(user_video_path)

    ref_total = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    user_total = int(cap_user.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_ref.get(cv2.CAP_PROP_FPS) or 24.0

    total_frames = min(ref_total, user_total)
    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Initialize MediaPipe Pose detectors
    pose_ref = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_user = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Read first frames to calculate dimensions
    ret_r, frame_r = cap_ref.read()
    ret_u, frame_u = cap_user.read()

    if not ret_r or not ret_u:
        raise ValueError("Could not read initial frames from videos.")

    cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_user.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Target height for video output
    target_h = 720
    new_ref_w = int(frame_r.shape[1] * (target_h / frame_r.shape[0]))
    new_user_w = int(frame_u.shape[1] * (target_h / frame_u.shape[0]))
    combined_w = new_ref_w + new_user_w
    combined_h = target_h + 110

    temp_raw_output = output_video_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_raw_output, fourcc, fps, (combined_w, combined_h))

    scores = []
    lead_elbow_diffs = []
    front_knee_diffs = []
    frame_metrics = []

    # Snapshots for each phase
    keyframe_snapshots = {}
    phase_targets = {
        "Stance & Setup": int(total_frames * 0.15),
        "Backlift & Stride": int(total_frames * 0.38),
        "Downswing & Impact": int(total_frames * 0.62),
        "Follow-Through & Balance": int(total_frames * 0.88),
    }

    frame_idx = 0
    while frame_idx < total_frames:
        ret_r, f_ref = cap_ref.read()
        ret_u, f_user = cap_user.read()

        if not ret_r or not ret_u:
            break

        progress = frame_idx / float(total_frames)
        phase = get_shot_phase(progress)

        # Pose extraction
        kp_ref, lm_ref = extract_landmarks(f_ref, pose_ref)
        kp_user, lm_user = extract_landmarks(f_user, pose_user)

        angles_ref = compute_cricket_angles(kp_ref)
        angles_user = compute_cricket_angles(kp_user)

        score, feedback, status = evaluate_frame_form(angles_user, angles_ref, phase)
        scores.append(score)

        if angles_user and angles_ref:
            lead_elbow_diffs.append(abs(angles_user["lead_elbow"] - angles_ref["lead_elbow"]))
            front_knee_diffs.append(abs(angles_user["front_knee"] - angles_ref["front_knee"]))
            frame_metrics.append({
                "frame": frame_idx,
                "phase": phase,
                "score": score,
                "user_lead_elbow": angles_user["lead_elbow"],
                "ref_lead_elbow": angles_ref["lead_elbow"],
                "user_front_knee": angles_user["front_knee"],
                "ref_front_knee": angles_ref["front_knee"],
                "feedback": feedback
            })

        # Draw overlays
        draw_skeleton_with_status(f_ref, lm_ref, status="ref", label="Pro")
        draw_skeleton_with_status(f_user, lm_user, status=status, label="User")

        # Combine
        combined_frame = create_side_by_side_frame(
            f_ref, f_user, angles_ref, angles_user,
            score, feedback, status, phase, target_h=target_h
        )

        out.write(combined_frame)

        # Save keyframe snapshots
        if snapshots_dir:
            for p_name, target_idx in phase_targets.items():
                if frame_idx == target_idx and p_name not in keyframe_snapshots:
                    snap_filename = f"snapshot_{p_name.replace(' ', '_').replace('&', 'and').lower()}.jpg"
                    snap_path = os.path.join(snapshots_dir, snap_filename)
                    cv2.imwrite(snap_path, combined_frame)
                    keyframe_snapshots[p_name] = snap_filename

        frame_idx += 1
        if progress_callback and frame_idx % 15 == 0:
            progress_callback(frame_idx, total_frames)

    cap_ref.release()
    cap_user.release()
    out.release()
    pose_ref.close()
    pose_user.close()

    # Transcode raw output to browser-playable H.264 MP4
    try:
        transcode_to_h264(temp_raw_output, output_video_path)
        if os.path.exists(temp_raw_output):
            os.remove(temp_raw_output)
    except Exception as e:
        print(f"H.264 transcode fallback: {e}")
        if os.path.exists(temp_raw_output) and not os.path.exists(output_video_path):
            os.rename(temp_raw_output, output_video_path)

    # Compile Final Coaching Report
    avg_score = round(float(np.mean(scores)), 1) if scores else 75.0
    avg_elbow_diff = round(float(np.mean(lead_elbow_diffs)), 1) if lead_elbow_diffs else 10.0
    avg_knee_diff = round(float(np.mean(front_knee_diffs)), 1) if front_knee_diffs else 12.0

    # Qualitative evaluation & drills
    strengths = []
    flaws = []
    drills = []

    if avg_elbow_diff <= 12.0:
        strengths.append("High Lead Elbow: Clean top-hand control guiding the shot along the ground.")
    else:
        flaws.append(f"Dropped Lead Elbow: Front elbow deviates by ~{avg_elbow_diff:.0f}°, risking aerial shots.")
        drills.append({
            "name": "High-Elbow Shadow Batting",
            "description": "Place a tennis ball under your chin and practice 30 shadow cover drives without dropping it to reinforce a high leading elbow."
        })

    if avg_knee_diff <= 14.0:
        strengths.append("Balanced Front Knee Bend: Solid base transferring body weight directly into the shot.")
    else:
        flaws.append(f"Front Leg Inconsistency: Front knee flexion differs by ~{avg_knee_diff:.0f}° from optimal drive depth.")
        drills.append({
            "name": "Front-Foot Lunge & Drive Drill",
            "description": "Perform front-foot lunges directly onto a marker cone, pausing at the point of impact to stabilize your knee flexion angle."
        })

    if avg_score >= 85:
        grade = "A (Master Class)"
        summary_verdict = "Exceptional cover drive execution! Your balance, lead elbow orientation, and swing line mirror professional standards."
    elif avg_score >= 75:
        grade = "B+ (Advanced)"
        summary_verdict = "Strong batting fundamentals! Minor adjustments to your lead elbow height will make your cover drives consistently grounded and authoritative."
    else:
        grade = "B (Developing Form)"
        summary_verdict = "Good foundational technique. Focus on leaning your head over the front knee and keeping the lead elbow high to prevent miscues."

    if not drills:
        drills.append({
            "name": "Drop-Ball Cover Drive Repetitions",
            "description": "Have a partner drop balls from shoulder height into your hitting zone to refine your timing and follow-through."
        })

    report = {
        "overall_score": avg_score,
        "grade": grade,
        "summary": summary_verdict,
        "lead_elbow_diff": avg_elbow_diff,
        "front_knee_diff": avg_knee_diff,
        "total_frames_analyzed": frame_idx,
        "strengths": strengths,
        "flaws": flaws,
        "drills": drills,
        "keyframes": keyframe_snapshots,
        "metrics_sample": frame_metrics[::10]  # Every 10th frame sample for chart rendering
    }

    return report
