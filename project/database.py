import os
import sqlite3
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sportiohub.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Athletes / Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS athletes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        sport TEXT NOT NULL,
        skill_level TEXT NOT NULL,
        location TEXT NOT NULL,
        role TEXT NOT NULL,
        bio TEXT,
        avatar TEXT,
        is_connected INTEGER DEFAULT 0
    )
    """)

    # Events table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        sport TEXT NOT NULL,
        event_type TEXT NOT NULL,
        date TEXT NOT NULL,
        location TEXT NOT NULL,
        organizer TEXT NOT NULL,
        description TEXT,
        max_participants INTEGER DEFAULT 50,
        registered_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'Open'
    )
    """)

    # Event Applications / Attendance table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS event_applications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id INTEGER NOT NULL,
        athlete_name TEXT NOT NULL,
        athlete_sport TEXT NOT NULL,
        skill_level TEXT NOT NULL,
        contact_email TEXT NOT NULL,
        notes TEXT,
        status TEXT DEFAULT 'Confirmed',
        applied_at TEXT NOT NULL,
        FOREIGN KEY (event_id) REFERENCES events (id)
    )
    """)

    # Community Posts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        author TEXT NOT NULL,
        author_sport TEXT NOT NULL,
        content TEXT NOT NULL,
        likes INTEGER DEFAULT 0,
        timestamp TEXT NOT NULL
    )
    """)

    # Pose Analysis History table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        title TEXT NOT NULL,
        similarity_score REAL,
        output_media TEXT,
        feedback TEXT,
        keyframes TEXT
    )
    """)

    conn.commit()

    # Seed initial data if empty
    cursor.execute("SELECT COUNT(*) FROM athletes")
    if cursor.fetchone()[0] == 0:
        seed_sample_data(conn)

    conn.close()

def seed_sample_data(conn):
    cursor = conn.cursor()

    # Initial Athletes
    sample_athletes = [
        ("Virat S.", "Cricket", "Advanced", "Delhi", "Top-Order Batsman", "Passionate top-order batsman focusing on cover drive mechanics and strike rotation.", "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150&auto=format&fit=crop&q=80", 1),
        ("Rohit K.", "Cricket", "Advanced", "Mumbai", "Opening Batsman", "Specializes in pull shot timing and power hitting in limited overs.", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&auto=format&fit=crop&q=80", 0),
        ("Jasprit B.", "Cricket", "Advanced", "Ahmedabad", "Pace Bowler", "Pace bowler working on yorker execution and seam presentation.", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150&auto=format&fit=crop&q=80", 0),
        ("Smriti M.", "Cricket", "Advanced", "Mumbai", "Left-Handed Opener", "Loves playing elegant drives through the off-side and cover region.", "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150&auto=format&fit=crop&q=80", 1),
        ("Sunil C.", "Football", "Advanced", "Bangalore", "Striker", "Forward player focusing on aerial headers, box positioning, and finishing.", "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&auto=format&fit=crop&q=80", 0),
        ("PV Sindhu", "Badminton", "Advanced", "Hyderabad", "Singles Pro", "Olympic medalist honing steep smashes and court coverage agility.", "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&auto=format&fit=crop&q=80", 1),
        ("Neeraj C.", "Athletics", "Advanced", "Haryana", "Javelin Thrower", "Focusing on core explosiveness and biomechanical arm speed.", "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=150&auto=format&fit=crop&q=80", 0),
        ("Rohan B.", "Tennis", "Intermediate", "Chennai", "All-Court Player", "Working on topspin backhands and baseline consistency.", "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=150&auto=format&fit=crop&q=80", 0)
    ]
    cursor.executemany("""
    INSERT INTO athletes (name, sport, skill_level, location, role, bio, avatar, is_connected)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_athletes)

    # Initial Events
    sample_events = [
        (
            "All-India T20 Cricket Championship 2026",
            "Cricket",
            "Tournament",
            "2026-11-15",
            "Wankhede Stadium, Mumbai",
            "Mumbai Sports Association",
            "Premier state tournament for open club teams. Scouts from state academies will be present.",
            64,
            42,
            "Open"
        ),
        (
            "Advanced Cricket Batting & Pose Masterclass",
            "Cricket",
            "Coaching Camp",
            "2026-10-25",
            "National Cricket Academy, Bangalore",
            "High Performance Coaching Staff",
            "Deep-dive biomechanical workshop covering high-elbow cover drives, balance over the crease, and stroke power.",
            30,
            24,
            "Open"
        ),
        (
            "State Youth Football Tryouts 2026",
            "Football",
            "Selection Trials",
            "2026-11-02",
            "Salt Lake Stadium, Kolkata",
            "Eastern Football Federation",
            "Annual selection trials for U-19 and open divisions. Looking for forwards, wingers, and central midfielders.",
            80,
            65,
            "Open"
        ),
        (
            "Premier Badminton Open Challenge",
            "Badminton",
            "Tournament",
            "2026-11-20",
            "Gachibowli Indoor Stadium, Hyderabad",
            "National Badminton League",
            "Men's and Women's singles and doubles tournament with graded rating points and equipment prizes.",
            40,
            28,
            "Open"
        ),
        (
            "Fast Bowling & Biomechanics Clinic",
            "Cricket",
            "Workshop",
            "2026-12-05",
            "Chepauk Ground, Chennai",
            "Tamil Nadu Sports Council",
            "Specialized video-analysis clinic for fast bowlers focusing on run-up deceleration, jump, and release angles.",
            25,
            18,
            "Open"
        )
    ]
    cursor.executemany("""
    INSERT INTO events (title, sport, event_type, date, location, organizer, description, max_participants, registered_count, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_events)

    # Initial Community Posts
    sample_posts = [
        ("Virat S.", "Cricket", "Just finished a 2-hour net session analyzing my cover drive with SportioHub Pose AI. Keeping the front elbow elevated made an instant difference!", 14, "2026-10-06 18:30"),
        ("Smriti M.", "Cricket", "Excited to attend the upcoming Masterclass in Bangalore! Connecting with other athletes here is such a game changer.", 8, "2026-10-05 14:15"),
        ("Sunil C.", "Football", "Trial season is approaching fast. Remember to build cardiovascular endurance alongside technical ball drills.", 21, "2026-10-04 09:45"),
        ("PV Sindhu", "Badminton", "Footwork speed is built through consistent lateral lunges. Looking forward to meeting everyone at the Hyderabad Open!", 19, "2026-10-03 11:20")
    ]
    cursor.executemany("""
    INSERT INTO posts (author, author_sport, content, likes, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """, sample_posts)

    conn.commit()

# --- Public API methods ---

def get_athletes(sport_filter=None, search=None):
    conn = get_connection()
    query = "SELECT * FROM athletes WHERE 1=1"
    params = []
    if sport_filter and sport_filter != "All":
        query += " AND sport = ?"
        params.append(sport_filter)
    if search:
        query += " AND (name LIKE ? OR role LIKE ? OR location LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])
    query += " ORDER BY is_connected DESC, id ASC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    athletes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return athletes

def toggle_connection(athlete_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT is_connected FROM athletes WHERE id = ?", (athlete_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    new_status = 0 if row["is_connected"] else 1
    cursor.execute("UPDATE athletes SET is_connected = ? WHERE id = ?", (new_status, athlete_id))
    conn.commit()
    conn.close()
    return bool(new_status)

def get_events(sport_filter=None):
    conn = get_connection()
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if sport_filter and sport_filter != "All":
        query += " AND sport = ?"
        params.append(sport_filter)
    query += " ORDER BY date ASC"

    cursor = conn.cursor()
    cursor.execute(query, params)
    events = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return events

def get_event_by_id(event_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    event = dict(row)

    # Fetch applicants
    cursor.execute("SELECT * FROM event_applications WHERE event_id = ? ORDER BY id DESC", (event_id,))
    applicants = [dict(r) for r in cursor.fetchall()]
    event["applicants"] = applicants
    conn.close()
    return event

def apply_for_event(event_id, athlete_name, athlete_sport, skill_level, contact_email, notes=""):
    conn = get_connection()
    cursor = conn.cursor()

    applied_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute("""
    INSERT INTO event_applications (event_id, athlete_name, athlete_sport, skill_level, contact_email, notes, status, applied_at)
    VALUES (?, ?, ?, ?, ?, ?, 'Confirmed', ?)
    """, (event_id, athlete_name, athlete_sport, skill_level, contact_email, notes, applied_at))

    # Increment registered_count
    cursor.execute("UPDATE events SET registered_count = registered_count + 1 WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()
    return True

def create_event(title, sport, event_type, date, location, organizer, description, max_participants=50):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO events (title, sport, event_type, date, location, organizer, description, max_participants, registered_count, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'Open')
    """, (title, sport, event_type, date, location, organizer, description, max_participants))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id

def get_posts():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts ORDER BY id DESC")
    posts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return posts

def create_post(author, author_sport, content):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute("""
    INSERT INTO posts (author, author_sport, content, likes, timestamp)
    VALUES (?, ?, ?, 0, ?)
    """, (author, author_sport, content, timestamp))
    post_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return post_id

def like_post(post_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE posts SET likes = likes + 1 WHERE id = ?", (post_id,))
    cursor.execute("SELECT likes FROM posts WHERE id = ?", (post_id,))
    row = cursor.fetchone()
    conn.commit()
    new_likes = row["likes"] if row else 0
    conn.close()
    return new_likes

def save_analysis_result(analysis_type, title, score, output_media, feedback_list, keyframes_dict=None):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    feedback_str = json.dumps(feedback_list) if isinstance(feedback_list, (list, dict)) else str(feedback_list)
    keyframes_str = json.dumps(keyframes_dict) if keyframes_dict else "{}"

    cursor.execute("""
    INSERT INTO analysis_history (timestamp, analysis_type, title, similarity_score, output_media, feedback, keyframes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, analysis_type, title, score, output_media, feedback_str, keyframes_str))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_analysis_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history ORDER BY id DESC")
    history = []
    for row in cursor.fetchall():
        item = dict(row)
        try:
            item["feedback"] = json.loads(item["feedback"])
        except Exception:
            item["feedback"] = [item["feedback"]]
        try:
            item["keyframes"] = json.loads(item["keyframes"])
        except Exception:
            item["keyframes"] = {}
        history.append(item)
    conn.close()
    return history
