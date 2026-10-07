# SportioHub

SportioHub is a Flask-based sports platform designed to support athletes with AI-driven coaching, form analysis, opportunities discovery, and a community landing page. The project combines computer vision, web interfaces, and a sports-focused user experience in a single application.

## Overview

This repository contains a web application under the `project/` directory that offers:

- Pose analysis for athletes using MediaPipe
- AI-style visual feedback for form correction
- Sports job and opportunity recommendations
- A community landing page and social-style posting UI
- Trainer/video comparison workflows

The app is built primarily in Python with Flask templates and static assets for the front end.

## Key Features

### 1. AI Pose Analysis
- Detects human pose landmarks using MediaPipe
- Compares input images against a validation image
- Calculates similarity and highlights deviations
- Suggests corrective feedback based on joint positions
- Saves output visuals for review

### 2. Sports Career / Opportunity Discovery
- User inputs skill level, preferred sport, and location
- Provides job recommendations and sports opportunities in a web UI
- Designed for athlete-focused career exploration

### 3. Community Experience
- Landing page includes carousel images and community posts
- Users can add new posts from the front end
- Social-style interface for engagement

### 4. Video Comparison Workflow
- Upload default and user exercise videos
- Perform pose-based comparison and overlay analysis
- Generate an output video with comparison feedback

## Project Structure

```text
SportioHub/
├── project/
│   ├── app.py
│   ├── pose_analysis.py
│   ├── comparison.mp4
│   ├── default.mp4
│   ├── static/
│   │   ├── images/
│   │   └── uploads/
│   ├── templates/
│   │   ├── ai.html
│   │   ├── chatbot.html
│   │   ├── display_video.html
│   │   ├── history.html
│   │   ├── index.html
│   │   ├── jobs.html
│   │   ├── login.html
│   │   └── trainer.html
│   └── ...
└── README.md
```

## Main Files

- `project/app.py` - Main Flask application and route definitions
- `project/pose_analysis.py` - MediaPipe-based landmark extraction and feedback logic
- `project/templates/` - HTML pages for Home, AI analysis, jobs, trainer UI, and more
- `project/static/` - Images, uploads, and static content

## Tech Stack

- Python
- Flask
- MediaPipe
- OpenCV
- Pillow (PIL)
- Matplotlib
- MongoDB Atlas (used in the analysis workflow)
- HTML / CSS / JavaScript / Tailwind

## Setup and Run

1. Navigate to the project folder:

```bash
cd project
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

3. Install dependencies:

```bash
pip install flask mediapipe opencv-python pillow matplotlib pymongo
```

4. Configure your environment if needed:
- Update the MongoDB connection string in `project/app.py` before using DB-backed routes.
- Ensure uploaded files and output directories are writable.

5. Start the Flask app:

```bash
python app.py
```

6. Open the app in your browser:

```text
http://localhost:5000/
```

## Routes

The application exposes several routes, including:

- `/` - Home page
- `/ai` - Pose comparison and analysis tool
- `/jobs` - Job and opportunity recommendations
- `/history` - Analysis history page
- `/trainer` - Video comparison workflow

## Example Use Cases

- Compare an athlete's pose against a reference image
- Receive actionable posture corrections
- Explore sports-related jobs or opportunities
- Browse a sports-themed community landing page

## Notes

This repository appears to include multiple prototype ideas combined into one app, including:

- pose analysis
- job recommendation UI
- a social/community page
- video-based comparison

It is a strong starting point for an athlete-focused AI sports platform, but some parts may need cleanup, dependency validation, and environment configuration before production use.

## License

This project does not currently include a license file. If needed, add one before distributing or deploying it publicly.

## Contributing

Contributions are welcome. To improve the project, consider:

- cleaning up duplicated legacy code in `app.py`
- adding proper dependency management via `requirements.txt`
- securing API keys and MongoDB configuration
- modularizing routes and utilities
- improving UI/UX and accessibility

---

Built for athlete performance analysis and sports career support.
