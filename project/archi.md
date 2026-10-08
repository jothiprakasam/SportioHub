# SportioHub Architecture Diagram

This document illustrates the complete architectural design of **SportioHub**, covering the front-end user experience, the Flask application backend, the AI/computer-vision pose analysis engine, and data persistence.

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    subgraph UI ["Client Layer (Browser / Frontend)"]
        A1["Home Feed & Community (/)\n• Training Updates\n• Post Creation & Likes"]
        A2["Athletes Network (/athletes)\n• Directory & Filters\n• Connect / Network"]
        A3["Events Hub (/events)\n• Tournament Discovery\n• Attend / Apply Modal\n• Host New Event"]
        A4["Cricket Pose Analyzer (/analyzer)\n• 1-Click Sample Videos\n• Custom Video Upload\n• Dual Player & Scorecard"]
        A5["Image Alignment (/ai)\n• Photo Deviation Markers"]
        A6["Career & AI Coach (/jobs & /chatbot)\n• Opportunity Matcher\n• Sports AI Assistant"]
    end

    subgraph Backend ["Application Controller Layer (Flask 3.0)"]
        B1["app.py (Routes & REST APIs)"]
        B2["Static File Delivery\n• H.264 Video Streams\n• Keyframe Snapshots"]
    end

    subgraph Engine ["Core Processing & AI Services"]
        C1["Cricket Pose Engine\n(cricket_pose_engine.py)\n• MediaPipe 3D Keypoints\n• Biomechanical Angles\n• Phase Classification\n• Side-by-Side Video Stitcher"]
        C2["Image Pose Comparator\n(pose_analysis.py)\n• Euclidean & Cosine Metrics\n• Deviation Annotator"]
        C3["Career Recommender\n(recommend_service.py)\n• Cosine Similarity\n• Government & Private Jobs"]
        C4["Sports Coach Assistant\n(chatbot_service.py)\n• Cricket Biomechanics Rules\n• Optional Gemini LLM API"]
        C5["FFmpeg Transcoder\n(imageio-ffmpeg)\n• Libx264 / yuv420p Conversion"]
    end

    subgraph Storage ["Data & Media Persistence Layer"]
        D1[("SQLite Database\nsportiohub.db")]
        D2[("Media Storage\n• static/outputs/\n• static/uploads/\n• default.mp4 / comparison.mp4")]
    end

    %% Connections
    A1 <--> B1
    A2 <--> B1
    A3 <--> B1
    A4 <--> B1
    A5 <--> B1
    A6 <--> B1

    B1 --> C1
    B1 --> C2
    B1 --> C3
    B1 --> C4

    C1 --> C5
    C5 --> D2
    C1 --> D2
    C2 --> D2

    B1 <--> D1
    B1 <--> D2
    B2 <--> D2
```

---

## 2. Cricket Pose Analyzer Video Pipeline

The video analysis pipeline takes either pre-loaded sample videos (`default.mp4` vs `comparison.mp4`) or custom athlete footage, processes every frame through MediaPipe Pose, evaluates cricket batting technique, and produces web-compatible side-by-side videos.

```mermaid
flowchart LR
    subgraph Inputs ["Video Input"]
        V1["Reference Video\n(default.mp4)"]
        V2["User Video\n(comparison.mp4)"]
    end

    subgraph PoseExtraction ["Frame & Landmark Extraction"]
        F1["Frame Synchronization\n(cv2.VideoCapture)"]
        MP["MediaPipe Pose AI\n(33 3D Keypoints)"]
    end

    subgraph Biomechanics ["Cricket Biomechanical Analysis"]
        BA1["Lead Elbow Angle\n[11-13-15] (High Elbow)"]
        BA2["Front Knee Flexion\n[23-25-27] (Lunge Depth)"]
        BA3["Head & Spine Alignment\n[Nose vs Front Knee]"]
        PH["Shot Phase Detection\n• Stance & Setup\n• Backlift & Stride\n• Impact Point\n• Follow-Through"]
    end

    subgraph Scoring ["Form Evaluation Engine"]
        SC["Real-time Angle Comparison\n& Form Score Calculation"]
        FB["Instant Coaching Tips\n& Qualitative Drill Generation"]
    end

    subgraph Rendering ["Side-by-Side Video & HUD Generation"]
        REN["Dual-Screen Stitcher\n(cv2.VideoWriter)\n• Reference Skeleton\n• User Skeleton & Alerts\n• Score Meter & Badges"]
        SNAP["Phase Keyframe Snapshots\n(4 Hi-Res Images)"]
        FF["FFmpeg Transcode\n(libx264, yuv420p)"]
    end

    subgraph Outputs ["Final Deliverables"]
        OUT1["Browser-Playable MP4 Video"]
        OUT2["Keyframe Snapshot Gallery"]
        OUT3["Coach Scorecard & Drills"]
    end

    V1 --> F1
    V2 --> F1
    F1 --> MP
    MP --> BA1
    MP --> BA2
    MP --> BA3
    MP --> PH

    BA1 --> SC
    BA2 --> SC
    BA3 --> SC
    PH --> SC

    SC --> FB
    SC --> REN
    FB --> REN
    REN --> SNAP
    REN --> FF
    FF --> OUT1
    SNAP --> OUT2
    FB --> OUT3
```

---

## 3. Athlete Networking & Events Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Athlete as Sports Person / Athlete
    participant UI as Web Frontend
    participant App as Flask Server (app.py)
    participant DB as SQLite (database.py)

    Note over Athlete, DB: 1. Athlete Discovery & Networking
    Athlete->>UI: Navigates to /athletes (filters by Sport/City)
    UI->>App: GET /athletes?sport=Cricket
    App->>DB: get_athletes(sport_filter="Cricket")
    DB-->>App: List of Athletes
    App-->>UI: Render Athlete Cards
    Athlete->>UI: Clicks "Connect" on an athlete card
    UI->>App: POST /api/athletes/{id}/connect
    App->>DB: toggle_connection(id)
    DB-->>App: is_connected = True
    App-->>UI: 200 OK (Badge toggles to "Connected")

    Note over Athlete, DB: 2. Sports Event Registration
    Athlete->>UI: Navigates to /events
    UI->>App: GET /events
    App->>DB: get_events()
    DB-->>App: List of Events with Registered Counts
    App-->>UI: Render Event Cards & Capacity Bars
    Athlete->>UI: Clicks "Attend / Apply Now"
    UI->>UI: Opens Registration Modal
    Athlete->>UI: Submits Name, Skill Level, Contact & Role
    UI->>App: POST /events/{id}/apply
    App->>DB: apply_for_event(id, athlete_data)
    DB-->>App: Confirmed & Increments Registered Count
    App-->>UI: Success Confirmation Toast
    UI-->>Athlete: Registration Confirmed!
```

---

## 4. Database Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    ATHLETES {
        int id PK
        string name
        string sport
        string skill_level
        string location
        string role
        string bio
        string avatar
        int is_connected
    }

    EVENTS {
        int id PK
        string title
        string sport
        string event_type
        string date
        string location
        string organizer
        string description
        int max_participants
        int registered_count
        string status
    }

    EVENT_APPLICATIONS {
        int id PK
        int event_id FK
        string athlete_name
        string athlete_sport
        string skill_level
        string contact_email
        string notes
        string status
        string applied_at
    }

    POSTS {
        int id PK
        string author
        string author_sport
        string content
        int likes
        string timestamp
    }

    ANALYSIS_HISTORY {
        int id PK
        string timestamp
        string analysis_type
        string title
        float similarity_score
        string output_media
        string feedback
        string keyframes
    }

    EVENTS ||--o{ EVENT_APPLICATIONS : "receives"
```

---

## 5. Technology Stack Summary

| Layer              | Technologies                                              | Purpose                                                                          |
| ------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Frontend**       | HTML5, Tailwind CSS, FontAwesome, JavaScript (Fetch/AJAX) | Responsive sports dark UI, video controls, modals, and real-time DOM updates     |
| **Backend**        | Python 3.12, Flask 3.0, Werkzeug                          | REST APIs, route controllers, static file serving, application management        |
| **Pose Analysis**  | Google MediaPipe Pose (`mediapipe`), OpenCV (`cv2`)       | 33 3D landmark extraction, biomechanical angle computation, skeleton rendering   |
| **Video Engine**   | `imageio-ffmpeg` (libx264, yuv420p)                       | Side-by-side stitcher and transcode to universally browser-compatible MP4        |
| **Recommendation** | Scikit-Learn (`cosine_similarity`), NumPy, Pandas         | Career and sports opportunities profile matching                                 |
| **Database**       | SQLite (`sportiohub.db`) with MongoDB Atlas fallback      | Persistent storage of athletes, events, applications, feed posts, and scorecards |
