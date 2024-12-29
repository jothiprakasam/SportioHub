import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

# Sample dataset generation for sports opportunities and users

# Constants
sports = ["Cricket", "Football", "Basketball", "Tennis", "Badminton"]
locations = ["North", "South", "East", "West", "Central"]
skills = ["Beginner", "Intermediate", "Advanced"]
opportunity_types = ["Training", "Tournaments", "Coaching", "Clubs"]
age_groups = ["Youth", "Adults", "All"]

# Generate random opportunities data (5 opportunities)
opportunities_data = [
    {"OpportunityID": 1, "Sport": "Cricket", "Location": "North", "SkillLevelRequired": "Beginner",
     "OpportunityType": "Training", "Duration": 2, "AgeGroup": "Youth"},
    {"OpportunityID": 2, "Sport": "Football", "Location": "South", "SkillLevelRequired": "Intermediate",
     "OpportunityType": "Tournaments", "Duration": 3, "AgeGroup": "Adults"},
    {"OpportunityID": 3, "Sport": "Basketball", "zLocation": "East", "SkillLevelRequired": "Advanced",
     "OpportunityType": "Coaching", "Duration": 1, "AgeGroup": "All"},
    {"OpportunityID": 4, "Sport": "Tennis", "Location": "West", "SkillLevelRequired": "Intermediate",
     "OpportunityType": "Clubs", "Duration": 4, "AgeGroup": "Youth"},
    {"OpportunityID": 5, "Sport": "Badminton", "Location": "Central", "SkillLevelRequired": "Advanced",
     "OpportunityType": "Training", "Duration": 2, "AgeGroup": "Adults"}
]

# Create DataFrame for opportunities
opportunities_df = pd.DataFrame(opportunities_data)

# Generate random user data (5 users)
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

# Create user DataFrame
users_df = pd.DataFrame(user_data)

# Encoding categorical features for both user and opportunities data
label_encoder = LabelEncoder()

# Encode user data
users_df['SkillLevelID'] = label_encoder.fit_transform(users_df['SkillLevel'])
users_df['LocationPreferenceID'] = label_encoder.fit_transform(users_df['LocationPreference'])
users_df['AgeGroupID'] = label_encoder.fit_transform(users_df['AgeGroup'])

# Encode opportunities data
opportunities_df['SportID'] = label_encoder.fit_transform(opportunities_df['Sport'])
opportunities_df['SkillLevelRequiredID'] = label_encoder.fit_transform(opportunities_df['SkillLevelRequired'])
opportunities_df['LocationID'] = label_encoder.fit_transform(opportunities_df['Location'])
opportunities_df['AgeGroupID'] = label_encoder.fit_transform(opportunities_df['AgeGroup'])


# Create a user-opportunity matching matrix (binary for sport preferences)
def create_sport_matrix(users_df, opportunities_df, sports):
    user_sport_matrix = np.zeros((len(users_df), len(sports)), dtype=int)
    opportunity_sport_matrix = np.zeros((len(opportunities_df), len(sports)), dtype=int)

    # Create user sport matrix (binary)
    for idx, row in users_df.iterrows():
        preferred_sports = row['PreferredSports'].split(', ')
        for sport in preferred_sports:
            if sport in sports:
                user_sport_matrix[idx, sports.index(sport)] = 1

    # Create opportunity sport matrix (binary)
    for idx, row in opportunities_df.iterrows():
        sport = row['Sport']
        if sport in sports:
            opportunity_sport_matrix[idx, sports.index(sport)] = 1

    return user_sport_matrix, opportunity_sport_matrix


# Sport matrix for matching users to opportunities
user_sport_matrix, opportunity_sport_matrix = create_sport_matrix(users_df, opportunities_df, sports)

# Cosine similarity between users and opportunities based on sports preferences
user_opportunity_similarity = cosine_similarity(user_sport_matrix, opportunity_sport_matrix)


# Function to recommend opportunities for a given user
def recommend_opportunities_for_user(user_id, user_opportunity_similarity, opportunities_df, top_n=5):
    user_idx = user_id - 1  # Adjust for 0-indexing
    similarity_scores = user_opportunity_similarity[user_idx]

    # Get the indices of the most similar opportunities
    top_opportunity_indices = similarity_scores.argsort()[-top_n:][::-1]

    recommendations = opportunities_df.iloc[top_opportunity_indices]
    recommendations['SimilarityScore'] = similarity_scores[top_opportunity_indices]

    return recommendations[[
        'OpportunityID', 'Sport', 'Location', 'SkillLevelRequired', 'OpportunityType', 'Duration', 'AgeGroup',
        'SimilarityScore']]


# Example: Recommend opportunities for User 3
recommended_opportunities = recommend_opportunities_for_user(3, user_opportunity_similarity, opportunities_df)

# Set display options to show all rows and columns if necessary
pd.set_option("display.max_rows", None)  # Display all rows
pd.set_option("display.max_columns", None)  # Display all columns
pd.set_option("display.width", None)  # No line truncation

print(recommended_opportunities)
