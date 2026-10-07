#!/usr/bin/env python3
"""
SportioHub Sports Opportunities & Job Recommendation CLI
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from recommend_service import recommend_jobs_and_opportunities

def main():
    print("=" * 60)
    print("🏆 SPORTIOHUB CAREER & OPPORTUNITIES RECOMMENDER")
    print("=" * 60)

    # Sample query
    sport = "Cricket"
    skill = "Advanced"
    location = "North"

    print(f"Searching recommendations for: Sport={sport}, Skill={skill}, Location={location}\n")
    results = recommend_jobs_and_opportunities(skill, sport, location)

    print("🏛️ GOVERNMENT SECTOR OPPORTUNITIES:")
    for job in results["governmentJobs"]:
        print(f"  • [{job['MatchScore']}% Match] {job['Title']} ({job['Organization']})")
        print(f"    Location: {job['Location']} | Type: {job['OpportunityType']} | Stipend: {job['Stipend']}")

    print("\n🏢 PRIVATE SECTOR OPPORTUNITIES:")
    for job in results["privateJobs"]:
        print(f"  • [{job['MatchScore']}% Match] {job['Title']} ({job['Organization']})")
        print(f"    Location: {job['Location']} | Type: {job['OpportunityType']} | Stipend: {job['Stipend']}")

    print("=" * 60)

if __name__ == "__main__":
    main()
