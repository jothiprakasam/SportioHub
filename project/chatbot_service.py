import os
import re

# Comprehensive knowledge base for Cricket & Athlete coaching
CRICKET_KNOWLEDGE = [
    {
        "keywords": ["cover drive", "elbow", "high elbow", "drive"],
        "answer": "🏏 **Mastering the Cover Drive & High Elbow**:\n\n1. **Lead with the Front Elbow**: The front elbow must point directly towards the target (cover / extra-cover). A high leading elbow ensures the bat swings cleanly along the ball's trajectory, keeping the shot along the ground.\n2. **Head Over the Ball**: Lean your head and front shoulder over the front knee. If your head falls back, the ball will fly into the air.\n3. **Front Knee Flexion**: Bend your front knee to roughly 125°-140° to transfer your body weight into the pitch of the ball.\n4. **Recommended Drill**: Practice shadow drives with a tennis ball tucked under your chin, or have a coach drop balls into your hitting zone."
    },
    {
        "keywords": ["pull shot", "hook", "back foot", "short ball"],
        "answer": "🏏 **Executing the Pull Shot Safely**:\n\n1. **Back & Across Stride**: Shift your weight onto your back foot inside the line of the short delivery.\n2. **Roll the Wrists**: Roll both wrists at the moment of impact to keep the ball grounded.\n3. **High to Low Swing**: Bring the bat from high to low rather than scooping upwards.\n4. **Eyes on the Seam**: Track the ball closely until it makes contact with the sweet spot."
    },
    {
        "keywords": ["bowling", "fast bowling", "pace", "seam", "run up"],
        "answer": "⚡ **Fast Bowling Biomechanics & Pace Generation**:\n\n1. **Smooth Acceleration**: Avoid stuttering in your run-up; build rhythmic momentum leading into the bound.\n2. **Back-Foot Landing**: Ensure your back foot lands securely to absorb impact and convert linear momentum into rotational velocity.\n3. **Non-Bowling Arm**: Pull your non-bowling arm down aggressively past your ribcage to accelerate your bowling shoulder.\n4. **Wrist Position**: Keep the wrist cocked behind the ball with the seam upright until final release."
    },
    {
        "keywords": ["fitness", "stamina", "workout", "gym", "speed", "agility"],
        "answer": "🏃 **Sports Conditioning & Agility Guidelines**:\n\n1. **Core Strength**: Planks, Russian twists, and medicine ball rotational throws develop rotational power for batting and throwing.\n2. **Leg Explosiveness**: Box jumps, Bulgarian split squats, and lateral lunges build strong deceleration and quick stride changes.\n3. **Sprint Intervals**: 6x50m shuttle sprints with 30s rest to simulate running between wickets or pitch sprints.\n4. **Active Recovery**: Static stretching and foam rolling post-training prevent hamstring and groin tightness."
    },
    {
        "keywords": ["event", "tournament", "trial", "apply", "attend"],
        "answer": "🏆 **Participating in Events on SportioHub**:\n\nHead to the **Events** tab on the navigation bar! You can browse upcoming state championships, masterclasses, and open selection trials. Click **'Attend / Apply'** on any event to register your profile and secure your slot."
    },
    {
        "keywords": ["pose", "analyze", "analyzer", "camera", "video"],
        "answer": "📸 **Using the SportioHub Pose Analyzer**:\n\n1. Open the **Pose Analyzer** page.\n2. You can either test the pre-loaded sample videos (`default.mp4` vs `comparison.mp4`) with 1-click, or upload your own batting videos.\n3. Our AI model extracts 33 body landmarks, tracks your lead elbow angle, knee flexion, and balance, then renders a side-by-side comparison video with coaching tips!"
    }
]

def query_sports_coach(prompt_text):
    """
    Intelligent sports coach response generator.
    Falls back to Gemini API if GEMINI_API_KEY is configured in the environment.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            system_instruction = "You are SportioHub AI, an elite Cricket and Sports Performance Coach. Give concise, encouraging, and technically sound advice for athletes."
            response = model.generate_content(f"{system_instruction}\n\nUser Question: {prompt_text}")
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"Gemini API fallback to local coach: {e}")

    # Local Knowledge Matcher
    p_lower = prompt_text.lower()
    for item in CRICKET_KNOWLEDGE:
        for kw in item["keywords"]:
            if re.search(r'\b' + re.escape(kw) + r'\b', p_lower):
                return item["answer"]

    # General encouraging coaching response
    return (
        "🏅 **SportioHub AI Coach Tip**:\n\n"
        f"Thank you for asking about *'{prompt_text[:60]}...'*\n\n"
        "Key athletic fundamentals to keep in mind:\n"
        "• **Balance & Base**: Great athletic performance begins with a solid, balanced stance.\n"
        "• **Repetition & Form**: Use the **Pose Analyzer** tab to check your posture deviations frame-by-frame.\n"
        "• **Community & Trials**: Connect with fellow athletes and coaches on our **Athletes Network** and attend upcoming **Events**!\n\n"
        "Feel free to ask specifically about the *cover drive*, *fast bowling*, *fitness drills*, or *how to use the video pose analyzer*!"
    )
