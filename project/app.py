#!/usr/bin/env python3
"""
SportioHub - Athlete Connect, Events & AI Cricket Pose Analysis Platform
"""

import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# Ensure project directory is in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import database
import cricket_pose_engine
import pose_analysis
from recommend_service import recommend_jobs_and_opportunities
from chatbot_service import query_sports_coach

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sportiohub-secret-key-2026'

# Configure upload and output folders inside static
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max

# Initialize database tables and initial sample data
database.init_db()

# -------------------------------------------------------------
# Home & Community Feed Routes
# -------------------------------------------------------------

@app.route('/')
def home():
    posts = database.get_posts()
    athletes = database.get_athletes()[:4]
    upcoming_events = database.get_events()[:3]
    return render_template(
        'index.html',
        posts=posts,
        athletes=athletes,
        events=upcoming_events
    )

@app.route('/api/posts', methods=['GET', 'POST'])
def api_posts():
    if request.method == 'GET':
        return jsonify(database.get_posts())

    data = request.get_json() or request.form
    author = data.get('author', 'Anonymous Athlete')
    author_sport = data.get('author_sport', 'Cricket')
    content = data.get('content', '').strip()

    if not content:
        return jsonify({'error': 'Post content cannot be empty.'}), 400

    post_id = database.create_post(author, author_sport, content)
    return jsonify({
        'message': 'Post created successfully!',
        'post_id': post_id,
        'post': {
            'id': post_id,
            'author': author,
            'author_sport': author_sport,
            'content': content,
            'likes': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    }), 201

@app.route('/api/posts/<int:post_id>/like', methods=['POST'])
def like_post_api(post_id):
    new_likes = database.like_post(post_id)
    return jsonify({'likes': new_likes})

# -------------------------------------------------------------
# Athletes Network & Connection Routes
# -------------------------------------------------------------

@app.route('/athletes')
def athletes_directory():
    sport_filter = request.args.get('sport', 'All')
    search_query = request.args.get('q', '').strip()
    athletes = database.get_athletes(sport_filter=sport_filter, search=search_query)
    return render_template(
        'athletes.html',
        athletes=athletes,
        current_sport=sport_filter,
        search_query=search_query
    )

@app.route('/api/athletes/<int:athlete_id>/connect', methods=['POST'])
def toggle_athlete_connect(athlete_id):
    status = database.toggle_connection(athlete_id)
    if status is None:
        return jsonify({'error': 'Athlete not found'}), 404
    return jsonify({'is_connected': status})

# -------------------------------------------------------------
# Sports Events & Registration Routes
# -------------------------------------------------------------

@app.route('/events')
def events_page():
    sport_filter = request.args.get('sport', 'All')
    events = database.get_events(sport_filter=sport_filter)
    return render_template('events.html', events=events, current_sport=sport_filter)

@app.route('/events/<int:event_id>')
def event_details(event_id):
    event = database.get_event_by_id(event_id)
    if not event:
        return redirect(url_for('events_page'))
    return render_template('event_detail.html', event=event)

@app.route('/events/<int:event_id>/apply', methods=['POST'])
def apply_event(event_id):
    name = request.form.get('athlete_name', '').strip()
    sport = request.form.get('athlete_sport', 'Cricket').strip()
    skill = request.form.get('skill_level', 'Intermediate').strip()
    email = request.form.get('contact_email', '').strip()
    notes = request.form.get('notes', '').strip()

    if not name or not email:
        return jsonify({'error': 'Name and Email are required'}), 400

    database.apply_for_event(event_id, name, sport, skill, email, notes)
    return jsonify({'message': f'Successfully registered {name} for this event!'})

@app.route('/events/new', methods=['POST'])
def create_new_event():
    title = request.form.get('title', '').strip()
    sport = request.form.get('sport', 'Cricket').strip()
    event_type = request.form.get('event_type', 'Tournament').strip()
    date = request.form.get('date', '').strip()
    location = request.form.get('location', '').strip()
    organizer = request.form.get('organizer', '').strip()
    description = request.form.get('description', '').strip()
    max_part = int(request.form.get('max_participants', 50))

    if not title or not date or not location:
        return jsonify({'error': 'Title, Date, and Location are required'}), 400

    event_id = database.create_event(title, sport, event_type, date, location, organizer, description, max_part)
    return jsonify({'message': 'Event created successfully!', 'event_id': event_id})

# -------------------------------------------------------------
# Cricket Pose Analyzer (Video Analysis) Routes
# -------------------------------------------------------------

@app.route('/trainer', methods=['GET', 'POST'])
@app.route('/analyzer', methods=['GET', 'POST'])
def trainer():
    if request.method == 'POST':
        action = request.form.get('action')

        # Mode A: Run pre-loaded sample videos
        if action == 'run_sample':
            default_vid = os.path.join(BASE_DIR, 'default.mp4')
            user_vid = os.path.join(BASE_DIR, 'comparison.mp4')
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_filename = f"cricket_sample_analysis_{run_id}.mp4"
            out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)

            report = cricket_pose_engine.analyze_cricket_videos(
                default_vid,
                user_vid,
                out_path,
                snapshots_dir=app.config['OUTPUT_FOLDER'],
                max_frames=120  # Fast high-fidelity analysis for responsive web preview
            )

            # Save in database history
            database.save_analysis_result(
                analysis_type="Cricket Batting Video (Sample)",
                title="Cover Drive Form Comparison (Sample Videos)",
                score=report['overall_score'],
                output_media=out_filename,
                feedback_list=report['flaws'] + report['strengths'],
                keyframes_dict=report['keyframes']
            )

            return render_template(
                'trainer.html',
                output_video=out_filename,
                report=report,
                analyzed=True,
                is_sample=True
            )

        # Mode B: Custom uploaded videos
        default_file = request.files.get('default_video')
        user_file = request.files.get('user_video')

        if not user_file or user_file.filename == '':
            return render_template('trainer.html', error="Please select your performance video.")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_vid_path = os.path.join(app.config['UPLOAD_FOLDER'], f"user_{run_id}_{secure_filename(user_file.filename)}")
        user_file.save(user_vid_path)

        # Use default reference if user did not provide reference video
        if default_file and default_file.filename != '':
            default_vid_path = os.path.join(app.config['UPLOAD_FOLDER'], f"ref_{run_id}_{secure_filename(default_file.filename)}")
            default_file.save(default_vid_path)
        else:
            default_vid_path = os.path.join(BASE_DIR, 'default.mp4')

        out_filename = f"cricket_user_analysis_{run_id}.mp4"
        out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)

        report = cricket_pose_engine.analyze_cricket_videos(
            default_vid_path,
            user_vid_path,
            out_path,
            snapshots_dir=app.config['OUTPUT_FOLDER'],
            max_frames=150
        )

        database.save_analysis_result(
            analysis_type="Cricket Batting Video (Custom)",
            title="Custom Batting Video Analysis",
            score=report['overall_score'],
            output_media=out_filename,
            feedback_list=report['flaws'] + report['strengths'],
            keyframes_dict=report['keyframes']
        )

        return render_template(
            'trainer.html',
            output_video=out_filename,
            report=report,
            analyzed=True,
            is_sample=False
        )

    return render_template('trainer.html')

# -------------------------------------------------------------
# Image Pose Analysis Route
# -------------------------------------------------------------

@app.route('/ai', methods=['GET', 'POST'])
def ai_analysis():
    if request.method == 'POST':
        action = request.form.get('action')

        # Option to run sample images
        if action == 'run_sample':
            input_path = os.path.join(BASE_DIR, 'upload.jpeg')
            val_path = os.path.join(BASE_DIR, 'test.jpeg')
            out_filename = f"output_sample_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_filename)

            k_in, img_in = pose_analysis.extract_keypoints(input_path)
            k_val, img_val = pose_analysis.extract_keypoints(val_path)

            if not k_in or not k_val:
                return render_template('ai.html', error="Failed to detect pose landmarks in sample images.")

            score = pose_analysis.calculate_similarity(k_in, k_val)
            devs = pose_analysis.identify_deviations(k_in, k_val)
            feedback = pose_analysis.suggest_corrections(devs)
            pose_analysis.draw_feedback(img_in, devs, k_in, out_path)

            database.save_analysis_result(
                analysis_type="Image Pose Analysis (Sample)",
                title="Stance & Impact Pose Comparison (Sample Images)",
                score=score,
                output_media=out_filename,
                feedback_list=feedback
            )

            return render_template(
                'ai.html',
                similarity_score=score,
                feedback=feedback,
                output_image=out_filename,
                deviations_count=len(devs)
            )

        # Uploaded images
        input_file = request.files.get('input_image')
        validation_file = request.files.get('validation_image')

        if not input_file or not validation_file:
            return render_template('ai.html', error="Please upload both input and reference validation images.")

        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{ts}.jpg")
        val_path = os.path.join(app.config['UPLOAD_FOLDER'], f"val_{ts}.jpg")
        out_filename = f"output_{ts}.jpg"
        out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_filename)

        input_file.save(input_path)
        validation_file.save(val_path)

        k_in, img_in = pose_analysis.extract_keypoints(input_path)
        k_val, img_val = pose_analysis.extract_keypoints(val_path)

        if not k_in or not k_val:
            return render_template('ai.html', error="Failed to detect pose landmarks in one or both images.")

        score = pose_analysis.calculate_similarity(k_in, k_val)
        devs = pose_analysis.identify_deviations(k_in, k_val)
        feedback = pose_analysis.suggest_corrections(devs)
        pose_analysis.draw_feedback(img_in, devs, k_in, out_path)

        database.save_analysis_result(
            analysis_type="Image Pose Analysis",
            title="Custom Image Pose Comparison",
            score=score,
            output_media=out_filename,
            feedback_list=feedback
        )

        return render_template(
            'ai.html',
            similarity_score=score,
            feedback=feedback,
            output_image=out_filename,
            deviations_count=len(devs)
        )

    return render_template('ai.html')

# -------------------------------------------------------------
# Jobs & Career Opportunities Routes
# -------------------------------------------------------------

@app.route('/jobs', methods=['GET', 'POST'])
def jobs():
    return render_template('jobs.html')

@app.route('/recommend_jobs', methods=['POST'])
def recommend_jobs_api():
    data = request.get_json() or {}
    skill_level = data.get('skillLevel')
    preferred_sport = data.get('preferredSport')
    location_pref = data.get('locationPreference')

    results = recommend_jobs_and_opportunities(skill_level, preferred_sport, location_pref)
    return jsonify(results)

# -------------------------------------------------------------
# Chatbot & Sports Coach Assistant
# -------------------------------------------------------------

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        data = request.get_json() or {}
        user_prompt = data.get('prompt', '').strip()

        if not user_prompt:
            return jsonify({'error': 'Please provide a question or prompt.'}), 400

        reply = query_sports_coach(user_prompt)
        return jsonify({'response': reply})

    return render_template('chatbot.html')

# -------------------------------------------------------------
# History & User Authentication Routes
# -------------------------------------------------------------

@app.route('/history')
def history():
    results = database.get_analysis_history()
    return render_template('history.html', results=results)

@app.route('/login')
def login():
    return render_template('login.html')

# -------------------------------------------------------------
# Static Media Serving
# -------------------------------------------------------------

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/uploads/<path:filename>')
def serve_upload_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting SportioHub on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)