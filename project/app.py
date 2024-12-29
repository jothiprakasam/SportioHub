'''from flask import Flask, render_template, request, jsonify
from datetime import datetime
import google.generativeai as gen_ai
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


from flask import Flask, render_template, request, jsonify
from datetime import datetime
import google.generativeai as gen_ai
import os

app = Flask(__name__)

# Initialize Google Generative AI
gemini_api_key = "IzaSyBPUDHMLjx_fSEtnITIf695gyBKM4NQfAIA"
gen_ai.configure(api_key=gemini_api_key)

def query_gemini(prompt):
    try:
        # Generate response from Gemini API
        response = gen_ai.generate_text(prompt=prompt, max_output_tokens=150)
        return response.text  # Assuming the response has the 'text' field
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")

posts = [
    {"content": "Excited to announce a new training event this weekend!", "author": "User123", "timestamp": "2024-12-28"},
]

@app.route('/')
def home():
    return render_template('index.html')  # Ensure the frontend HTML is placed in the 'templates' folder

@app.route('/api/posts', methods=['GET', 'POST'])
def handle_posts():
    if request.method == 'GET':
        return jsonify(posts)

    if request.method == 'POST':
        data = request.json
        new_post = {
            "content": data.get("content"),
            "author": data.get("author", "Anonymous"),
            "timestamp": datetime.now().strftime("%Y-%m-%d")
        }
        posts.append(new_post)
        return jsonify({"message": "Post added successfully!", "post": new_post}), 201

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        # Render the chatbot page for the user to interact with
        return render_template('chatbot.html')

    if request.method == 'POST':
        # Get the user's prompt from the JSON body
        user_prompt = request.json.get('prompt')

        if not user_prompt:
            return jsonify({"error": "No prompt provided"}), 400

        try:
            # Get the chatbot's response from the Gemini API
            reply = query_gemini(user_prompt)
            return jsonify({"response": reply})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, render_template, request, jsonify
from datetime import datetime
import os
from pose_analysis import extract_keypoints, calculate_similarity, identify_deviations, suggest_corrections, draw_feedback
from pymongo import MongoClient
from PIL import Image

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB Atlas connection
MONGO_URI = "your_mongodb_atlas_connection_string"
client = MongoClient(MONGO_URI)
db = client['pose_analysis_db']  # Replace with your database name
results_collection = db['analysis_results']  # Replace with your collection name

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for AI analysis
@app.route('/ai', methods=['GET', 'POST'])
def ai_analysis():
    if request.method == 'POST':
        # Save uploaded input and validation images
        input_file = request.files.get('input_image')
        validation_file = request.files.get('validation_image')

        if not input_file or not validation_file:
            return render_template('ai.html', error="Please upload both input and validation images.")

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        validation_path = os.path.join(app.config['UPLOAD_FOLDER'], f"validation_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        
        input_file.save(input_path)
        validation_file.save(validation_path)

        # Analyze pose
        input_keypoints, input_image = extract_keypoints(input_path)
        validation_keypoints, validation_image = extract_keypoints(validation_path)

        if not input_keypoints or not validation_keypoints:
            return render_template('ai.html', error="Failed to detect pose landmarks in one or both images.")

        similarity_score = calculate_similarity(input_keypoints, validation_keypoints)
        deviations = identify_deviations(input_keypoints, validation_keypoints)
        feedback = suggest_corrections(deviations)

        # Generate feedback visualization
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        draw_feedback(input_image, deviations, input_keypoints, output_path)

        # Save results to MongoDB
        result = {
            "timestamp": datetime.now(),
            "input_image_path": input_path,
            "validation_image_path": validation_path,
            "output_image_path": output_path,
            "similarity_score": similarity_score,
            "deviations": deviations,
            "feedback": feedback
        }
        results_collection.insert_one(result)

        return render_template(
            'ai.html',
            similarity_score=similarity_score,
            feedback=feedback,
            output_image=output_path
        )

    return render_template('ai.html')

# Route to view analysis history
@app.route('/history')
def history():
    # Fetch all analysis results from MongoDB
    results = results_collection.find().sort("timestamp", -1)
    return render_template('history.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

#new






''' main
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Flask App Initialization
app = Flask(__name__)

# Data Initialization
sports = ["Cricket", "Football", "Basketball", "Tennis", "Badminton"]
locations = ["North", "South", "East", "West", "Central"]
skills = ["Beginner", "Intermediate", "Advanced"]
opportunity_types = ["Training", "Tournaments", "Coaching", "Clubs"]
age_groups = ["Youth", "Adults", "All"]

# Opportunities Data
opportunities_data = [
    {"OpportunityID": 1, "Sport": "Cricket", "Location": "North", "SkillLevelRequired": "Beginner",
     "OpportunityType": "Training", "Duration": 2, "AgeGroup": "Youth"},
    {"OpportunityID": 2, "Sport": "Football", "Location": "South", "SkillLevelRequired": "Intermediate",
     "OpportunityType": "Tournaments", "Duration": 3, "AgeGroup": "Adults"},
    {"OpportunityID": 3, "Sport": "Basketball", "Location": "East", "SkillLevelRequired": "Advanced",
     "OpportunityType": "Coaching", "Duration": 1, "AgeGroup": "All"},
    {"OpportunityID": 4, "Sport": "Tennis", "Location": "West", "SkillLevelRequired": "Intermediate",
     "OpportunityType": "Clubs", "Duration": 4, "AgeGroup": "Youth"},
    {"OpportunityID": 5, "Sport": "Badminton", "Location": "Central", "SkillLevelRequired": "Advanced",
     "OpportunityType": "Training", "Duration": 2, "AgeGroup": "Adults"}
]
government_jobs = [
    {"title": "National Coach", "location": "Delhi", "sport": "Cricket", "skillLevel": "Advanced", "type": "Full-Time"},
    {"title": "Assistant Trainer", "location": "Chennai", "sport": "Hockey", "skillLevel": "Intermediate", "type": "Contract"}
]

private_jobs = [
    {"title": "Fitness Coach", "location": "Bangalore", "sport": "Tennis", "skillLevel": "Beginner", "type": "Part-Time"},
    {"title": "Physiotherapist", "location": "Mumbai", "sport": "Football", "skillLevel": "Advanced", "type": "Full-Time"}
]
opportunities_df = pd.DataFrame(opportunities_data)

# Users Data
user_data = [
    {"UserID": 1, "PreferredSports": "Cricket, Football", "SkillLevel": "Beginner", "LocationPreference": "North",
     "AgeGroup": "Youth"},
    {"UserID": 2, "PreferredSports": "Basketball, Tennis", "SkillLevel": "Intermediate", "LocationPreference": "East",
     "AgeGroup": "Adults"},
    {"UserID": 3, "PreferredSports": "Football, Basketball", "SkillLevel": "Advanced", "LocationPreference": "South",
     "AgeGroup": "All"},
    {"UserID": 4, "PreferredSports": "Tennis, Badminton", "SkillLevel": "Intermediate", "LocationPreference": "West",
     "AgeGroup": "Youth"},
    {"UserID": 5, "PreferredSports": "Cricket, Badminton", "SkillLevel": "Advanced", "LocationPreference": "Central",
     "AgeGroup": "Adults"}
]

users_df = pd.DataFrame(user_data)

# Encoding Categorical Features
label_encoder = LabelEncoder()

users_df['SkillLevelID'] = label_encoder.fit_transform(users_df['SkillLevel'])
users_df['LocationPreferenceID'] = label_encoder.fit_transform(users_df['LocationPreference'])
users_df['AgeGroupID'] = label_encoder.fit_transform(users_df['AgeGroup'])

opportunities_df['SportID'] = label_encoder.fit_transform(opportunities_df['Sport'])
opportunities_df['SkillLevelRequiredID'] = label_encoder.fit_transform(opportunities_df['SkillLevelRequired'])
opportunities_df['LocationID'] = label_encoder.fit_transform(opportunities_df['Location'])
opportunities_df['AgeGroupID'] = label_encoder.fit_transform(opportunities_df['AgeGroup'])

# Create Sport Matrix
def create_sport_matrix(users_df, opportunities_df, sports):
    user_sport_matrix = np.zeros((len(users_df), len(sports)), dtype=int)
    opportunity_sport_matrix = np.zeros((len(opportunities_df), len(sports)), dtype=int)

    for idx, row in users_df.iterrows():
        preferred_sports = row['PreferredSports'].split(', ')
        for sport in preferred_sports:
            if sport in sports:
                user_sport_matrix[idx, sports.index(sport)] = 1

    for idx, row in opportunities_df.iterrows():
        sport = row['Sport']
        if sport in sports:
            opportunity_sport_matrix[idx, sports.index(sport)] = 1

    return user_sport_matrix, opportunity_sport_matrix

user_sport_matrix, opportunity_sport_matrix = create_sport_matrix(users_df, opportunities_df, sports)
user_opportunity_similarity = cosine_similarity(user_sport_matrix, opportunity_sport_matrix)
@app.route('/ai', methods=['GET', 'POST'])
def ai_analysis():
    if request.method == 'POST':
        # Save uploaded input and validation images
        input_file = request.files.get('input_image')
        validation_file = request.files.get('validation_image')

        if not input_file or not validation_file:
            return render_template('ai.html', error="Please upload both input and validation images.")

        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        validation_path = os.path.join(app.config['UPLOAD_FOLDER'], f"validation_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        
        input_file.save(input_path)
        validation_file.save(validation_path)

        # Analyze pose
        input_keypoints, input_image = extract_keypoints(input_path)
        validation_keypoints, validation_image = extract_keypoints(validation_path)

        if not input_keypoints or not validation_keypoints:
            return render_template('ai.html', error="Failed to detect pose landmarks in one or both images.")

        similarity_score = calculate_similarity(input_keypoints, validation_keypoints)
        deviations = identify_deviations(input_keypoints, validation_keypoints)
        feedback = suggest_corrections(deviations)

        # Generate feedback visualization
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
        draw_feedback(input_image, deviations, input_keypoints, output_path)

        return render_template(
            'ai.html',
            similarity_score=similarity_score,
            feedback=feedback,
            output_image=output_path
        )

    return render_template('ai.html')
@app.route('/recommend_jobs', methods=['POST'])
def recommend_jobs():
    try:
        data = request.get_json()
        skill_level = data.get('skillLevel')
        preferred_sport = data.get('preferredSport')
        location_preference = data.get('locationPreference')

        # Filter jobs based on user input
        filtered_government_jobs = [job for job in government_jobs
                                    if job["skillLevel"] == skill_level and job["sport"] == preferred_sport]
        filtered_private_jobs = [job for job in private_jobs
                                 if job["skillLevel"] == skill_level and job["sport"] == preferred_sport]

        # Respond with filtered jobs
        return jsonify({
            "governmentJobs": filtered_government_jobs,
            "privateJobs": filtered_private_jobs
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500


# Recommend Opportunities
def recommend_opportunities_for_user(user_id, user_opportunity_similarity, opportunities_df, top_n=5):
    user_idx = user_id - 1
    similarity_scores = user_opportunity_similarity[user_idx]
    top_opportunity_indices = similarity_scores.argsort()[-top_n:][::-1]

    recommendations = opportunities_df.iloc[top_opportunity_indices]
    recommendations['SimilarityScore'] = similarity_scores[top_opportunity_indices]

    return recommendations[[
        'OpportunityID', 'Sport', 'Location', 'SkillLevelRequired', 'OpportunityType', 'Duration', 'AgeGroup',
        'SimilarityScore']]

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/jobs', methods=['GET', 'POST'])
def jobs():
    if request.method == 'POST':
        # Process the POST request and return recommendations
        user_input = request.json
        # Extract user preferences from the input
        skill_level = user_input.get('skill_level')
        preferred_sport = user_input.get('preferred_sport')
        location = user_input.get('location')

        # Use the recommendation function to get job listings
        recommended_jobs = recommend_opportunities(skill_level, preferred_sport, location)
        return jsonify(recommended_jobs)

    # If it's a GET request, simply render the jobs page
    return render_template('jobs.html')


if __name__ == '__main__':
    app.run(debug=True)
'''

'''import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Flask App Initialization
app = Flask(__name__)

# Folder for uploading and saving videos
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Flask App Initialization
app = Flask(__name__)

# Data Initialization
sports = ["Cricket", "Football", "Basketball", "Tennis", "Badminton"]
locations = ["North", "South", "East", "West", "Central"]
skills = ["Beginner", "Intermediate", "Advanced"]
opportunity_types = ["Training", "Tournaments", "Coaching", "Clubs"]
age_groups = ["Youth", "Adults", "All"]

# Opportunities Data
opportunities_data = [
    {"OpportunityID": 1, "Sport": "Cricket", "Location": "North", "SkillLevelRequired": "Beginner",
     "OpportunityType": "Training", "Duration": 2, "AgeGroup": "Youth"},
    {"OpportunityID": 2, "Sport": "Football", "Location": "South", "SkillLevelRequired": "Intermediate",
     "OpportunityType": "Tournaments", "Duration": 3, "AgeGroup": "Adults"},
    {"OpportunityID": 3, "Sport": "Basketball", "Location": "East", "SkillLevelRequired": "Advanced",
     "OpportunityType": "Coaching", "Duration": 1, "AgeGroup": "All"},
    {"OpportunityID": 4, "Sport": "Tennis", "Location": "West", "SkillLevelRequired": "Intermediate",
     "OpportunityType": "Clubs", "Duration": 4, "AgeGroup": "Youth"},
    {"OpportunityID": 5, "Sport": "Badminton", "Location": "Central", "SkillLevelRequired": "Advanced",
     "OpportunityType": "Training", "Duration": 2, "AgeGroup": "Adults"}
]
government_jobs = [
    {"title": "National Coach", "location": "Delhi", "sport": "Cricket", "skillLevel": "Advanced", "type": "Full-Time"},
    {"title": "Assistant Trainer", "location": "Chennai", "sport": "Hockey", "skillLevel": "Intermediate", "type": "Contract"}
]

private_jobs = [
    {"title": "Fitness Coach", "location": "Bangalore", "sport": "Tennis", "skillLevel": "Beginner", "type": "Part-Time"},
    {"title": "Physiotherapist", "location": "Mumbai", "sport": "Football", "skillLevel": "Advanced", "type": "Full-Time"}
]
opportunities_df = pd.DataFrame(opportunities_data)

# Users Data
user_data = [
    {"UserID": 1, "PreferredSports": "Cricket, Football", "SkillLevel": "Beginner", "LocationPreference": "North",
     "AgeGroup": "Youth"},
    {"UserID": 2, "PreferredSports": "Basketball, Tennis", "SkillLevel": "Intermediate", "LocationPreference": "East",
     "AgeGroup": "Adults"},
    {"UserID": 3, "PreferredSports": "Football, Basketball", "SkillLevel": "Advanced", "LocationPreference": "South",
     "AgeGroup": "All"},
    {"UserID": 4, "PreferredSports": "Tennis, Badminton", "SkillLevel": "Intermediate", "LocationPreference": "West",
     "AgeGroup": "Youth"},
    {"UserID": 5, "PreferredSports": "Cricket, Badminton", "SkillLevel": "Advanced", "LocationPreference": "Central",
     "AgeGroup": "Adults"}
]

users_df = pd.DataFrame(user_data)

# Encoding Categorical Features
label_encoder = LabelEncoder()

users_df['SkillLevelID'] = label_encoder.fit_transform(users_df['SkillLevel'])
users_df['LocationPreferenceID'] = label_encoder.fit_transform(users_df['LocationPreference'])
users_df['AgeGroupID'] = label_encoder.fit_transform(users_df['AgeGroup'])

opportunities_df['SportID'] = label_encoder.fit_transform(opportunities_df['Sport'])
opportunities_df['SkillLevelRequiredID'] = label_encoder.fit_transform(opportunities_df['SkillLevelRequired'])
opportunities_df['LocationID'] = label_encoder.fit_transform(opportunities_df['Location'])
opportunities_df['AgeGroupID'] = label_encoder.fit_transform(opportunities_df['AgeGroup'])

# Create Sport Matrix
def create_sport_matrix(users_df, opportunities_df, sports):
    user_sport_matrix = np.zeros((len(users_df), len(sports)), dtype=int)
    opportunity_sport_matrix = np.zeros((len(opportunities_df), len(sports)), dtype=int)

    for idx, row in users_df.iterrows():
        preferred_sports = row['PreferredSports'].split(', ')
        for sport in preferred_sports:
            if sport in sports:
                user_sport_matrix[idx, sports.index(sport)] = 1

    for idx, row in opportunities_df.iterrows():
        sport = row['Sport']
        if sport in sports:
            opportunity_sport_matrix[idx, sports.index(sport)] = 1

    return user_sport_matrix, opportunity_sport_matrix

user_sport_matrix, opportunity_sport_matrix = create_sport_matrix(users_df, opportunities_df, sports)
user_opportunity_similarity = cosine_similarity(user_sport_matrix, opportunity_sport_matrix)

@app.route('/recommend_jobs', methods=['POST'])
def recommend_jobs():
    try:
        data = request.get_json()
        skill_level = data.get('skillLevel')
        preferred_sport = data.get('preferredSport')
        location_preference = data.get('locationPreference')

        # Filter jobs based on user input
        filtered_government_jobs = [job for job in government_jobs
                                    if job["skillLevel"] == skill_level and job["sport"] == preferred_sport]
        filtered_private_jobs = [job for job in private_jobs
                                 if job["skillLevel"] == skill_level and job["sport"] == preferred_sport]

        # Respond with filtered jobs
        return jsonify({
            "governmentJobs": filtered_government_jobs,
            "privateJobs": filtered_private_jobs
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500


# Recommend Opportunities
def recommend_opportunities_for_user(user_id, user_opportunity_similarity, opportunities_df, top_n=5):
    user_idx = user_id - 1
    similarity_scores = user_opportunity_similarity[user_idx]
    top_opportunity_indices = similarity_scores.argsort()[-top_n:][::-1]

    recommendations = opportunities_df.iloc[top_opportunity_indices]
    recommendations['SimilarityScore'] = similarity_scores[top_opportunity_indices]

    return recommendations[[
        'OpportunityID', 'Sport', 'Location', 'SkillLevelRequired', 'OpportunityType', 'Duration', 'AgeGroup',
        'SimilarityScore']]

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/jobs', methods=['GET', 'POST'])
def jobs():
    if request.method == 'POST':
        # Process the POST request and return recommendations
        user_input = request.json
        # Extract user preferences from the input
        skill_level = user_input.get('skill_level')
        preferred_sport = user_input.get('preferred_sport')
        location = user_input.get('location')

        # Use the recommendation function to get job listings
        recommended_jobs = recommend_opportunities(skill_level, preferred_sport, location)
        return jsonify(recommended_jobs)

    # If it's a GET request, simply render the jobs page
    return render_template('jobs.html')
# MediaPipe Initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Video Analysis Functions
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

    # Video Writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    while cap_default.isOpened() and cap_user.isOpened():
        ret_default, frame_default = cap_default.read()
        ret_user, frame_user = cap_user.read()

        if not ret_default or not ret_user:
            break

        # Extract keypoints and landmarks
        default_keypoints, default_landmarks = extract_keypoints(frame_default)
        user_keypoints, user_landmarks = extract_keypoints(frame_user)

        # Compare right elbow angles
        comparison_text = ""
        feedback_text = ""
        if default_keypoints is not None and user_keypoints is not None:
            default_angle = calculate_angle(default_keypoints[12], default_keypoints[14], default_keypoints[16])
            user_angle = calculate_angle(user_keypoints[12], user_keypoints[14], user_keypoints[16])
            angle_diff = abs(default_angle - user_angle)
            comparison_text = f"Right Elbow Angle Diff: {angle_diff:.2f}°"

            # Generate feedback based on the angle difference
            feedback_text = generate_feedback(angle_diff)

        # Overlay analysis on user frame
        frame_user = overlay_analysis(frame_user, user_landmarks, comparison_text)
        cv2.putText(frame_user, feedback_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Write the user video frame with analysis to the output video
        out.write(frame_user)

    cap_default.release()
    cap_user.release()
    out.release()

@app.route('/trainer', methods=['GET', 'POST'])
def trainer():
    if request.method == 'POST':
        if 'user_video' not in request.files:
            return "No file uploaded", 400

        user_video = request.files['user_video']
        user_video_path = os.path.join(app.config['UPLOAD_FOLDER'], user_video.filename)
        user_video.save(user_video_path)

        default_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'default_shot.mp4')
        output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], 'comparison_output.mp4')

        # Ensure default video exists
        if not os.path.exists(default_video_path):
            return "Default video not found", 500

        analyze_videos(default_video_path, user_video_path, output_video_path)

        return jsonify({"output_video": f"/outputs/comparison_output.mp4"})

    return render_template('trainer.html')

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
'''