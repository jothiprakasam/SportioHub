import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Pre-defined sports opportunities dataset
OPPORTUNITIES = [
    {
        "OpportunityID": 1,
        "Title": "National Junior Cricket Academy Trainee",
        "Sport": "Cricket",
        "Location": "North",
        "SkillLevelRequired": "Beginner",
        "OpportunityType": "Training",
        "Duration": "6 Months",
        "Organization": "Sports Authority of India (SAI)",
        "Sector": "Government",
        "Stipend": "₹25,000 / month"
    },
    {
        "OpportunityID": 2,
        "Title": "State League Football Forward",
        "Sport": "Football",
        "Location": "South",
        "SkillLevelRequired": "Intermediate",
        "OpportunityType": "Tournaments",
        "Duration": "1 Year",
        "Organization": "Premier Football Club",
        "Sector": "Private",
        "Stipend": "₹45,000 / month"
    },
    {
        "OpportunityID": 3,
        "Title": "High Performance Cricket Batting Coach",
        "Sport": "Cricket",
        "Location": "Central",
        "SkillLevelRequired": "Advanced",
        "OpportunityType": "Coaching",
        "Duration": "2 Years",
        "Organization": "State Cricket Board",
        "Sector": "Government",
        "Stipend": "₹80,000 / month"
    },
    {
        "OpportunityID": 4,
        "Title": "Junior Tennis Development Squad",
        "Sport": "Tennis",
        "Location": "West",
        "SkillLevelRequired": "Intermediate",
        "OpportunityType": "Clubs",
        "Duration": "8 Months",
        "Organization": "Elite Tennis Academy",
        "Sector": "Private",
        "Stipend": "₹35,000 / month"
    },
    {
        "OpportunityID": 5,
        "Title": "National Badminton Coaching Fellow",
        "Sport": "Badminton",
        "Location": "South",
        "SkillLevelRequired": "Advanced",
        "OpportunityType": "Training",
        "Duration": "1 Year",
        "Organization": "National Badminton Academy",
        "Sector": "Government",
        "Stipend": "₹60,000 / month"
    },
    {
        "OpportunityID": 6,
        "Title": "District Sports Officer (Cricket & Athletics)",
        "Sport": "Cricket",
        "Location": "North",
        "SkillLevelRequired": "Advanced",
        "OpportunityType": "Government Official",
        "Duration": "Permanent",
        "Organization": "Ministry of Youth Affairs & Sports",
        "Sector": "Government",
        "Stipend": "₹75,000 / month"
    },
    {
        "OpportunityID": 7,
        "Title": "Grassroots Football Scout & Analyst",
        "Sport": "Football",
        "Location": "East",
        "SkillLevelRequired": "Beginner",
        "OpportunityType": "Clubs",
        "Duration": "6 Months",
        "Organization": "City Football Club",
        "Sector": "Private",
        "Stipend": "₹30,000 / month"
    },
    {
        "OpportunityID": 8,
        "Title": "Assistant Tennis Coach & Sparring Partner",
        "Sport": "Tennis",
        "Location": "South",
        "SkillLevelRequired": "Intermediate",
        "OpportunityType": "Coaching",
        "Duration": "1 Year",
        "Organization": "International Racquet Club",
        "Sector": "Private",
        "Stipend": "₹40,000 / month"
    }
]

def recommend_jobs_and_opportunities(skill_level=None, preferred_sport=None, location_pref=None):
    """
    Filter and score opportunities based on user inputs.
    """
    gov_jobs = []
    priv_jobs = []

    for opp in OPPORTUNITIES:
        match_score = 60  # baseline

        if preferred_sport and opp["Sport"].lower() == preferred_sport.lower():
            match_score += 25
        if skill_level and opp["SkillLevelRequired"].lower() == skill_level.lower():
            match_score += 15
        if location_pref and location_pref.lower() in opp["Location"].lower():
            match_score += 10

        item = dict(opp)
        item["MatchScore"] = min(99, match_score)

        if opp["Sector"] == "Government":
            gov_jobs.append(item)
        else:
            priv_jobs.append(item)

    # Sort by match score descending
    gov_jobs.sort(key=lambda x: x["MatchScore"], reverse=True)
    priv_jobs.sort(key=lambda x: x["MatchScore"], reverse=True)

    return {
        "governmentJobs": gov_jobs,
        "privateJobs": priv_jobs
    }
